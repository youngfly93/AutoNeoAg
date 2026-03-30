#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoneoag.config import load_settings
from autoneoag.dataset import load_processed_dataset, split_frame
from autoneoag.metrics.ranking import metric_bundle


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: idx + 1 for idx, aa in enumerate(AMINO_ACIDS)}


@dataclass
class TrainConfig:
    max_peptide_len: int = 11
    max_hla_len: int = 34
    embed_dim: int = 16
    hidden_dim: int = 32
    dropout: float = 0.15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 40
    batch_size: int = 8
    focal_gamma: float = 1.5


def device_for_run() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode_sequence(sequence: str, length: int) -> list[int]:
    ids = [AA_TO_ID.get(char, 0) for char in sequence[:length]]
    return ids + [0] * (length - len(ids))


class NeoantigenRanker(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(len(AA_TO_ID) + 1, cfg.embed_dim)
        self.scalar_proj = nn.Sequential(
            nn.Linear(10, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.embed_dim * 4 + cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def _pool(self, tensor: torch.Tensor) -> torch.Tensor:
        mask = (tensor != 0).unsqueeze(-1)
        embedded = self.embedding(tensor)
        summed = (embedded * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1)
        return summed / denom

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        pooled = torch.cat(
            [
                self._pool(batch["mut_tokens"]),
                self._pool(batch["wt_tokens"]),
                self._pool(batch["hla_tokens"]),
                self._pool(batch["delta_tokens"]),
                self.scalar_proj(batch["scalars"]),
            ],
            dim=1,
        )
        return self.head(pooled).squeeze(1)


def focal_bce(logits: torch.Tensor, labels: torch.Tensor, gamma: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(labels > 0.5, probs, 1 - probs)
    return ((1 - pt) ** gamma * bce).mean()


def build_arrays(df) -> dict[str, np.ndarray]:
    mut = np.array([encode_sequence(item, 11) for item in df["peptide_mut"]], dtype=np.int64)
    wt = np.array([encode_sequence(item, 11) for item in df["peptide_wt"]], dtype=np.int64)
    hla = np.array([encode_sequence(item, 34) for item in df["hla_pseudosequence"]], dtype=np.int64)
    delta = np.array(
        [
            encode_sequence("".join(mc if mc != wc else "A" for mc, wc in zip(m, w, strict=True)), 11)
            for m, w in zip(df["peptide_mut"], df["peptide_wt"], strict=True)
        ],
        dtype=np.int64,
    )
    scalars = df[
        [
            "ba_score",
            "el_score",
            "ba_rank",
            "el_rank",
            "stab_score",
            "stab_rank",
            "gravy",
            "aromaticity",
            "non_polar_ratio",
            "agretopicity",
        ]
    ].fillna(0.0).to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy(dtype=np.float32)
    return {"mut": mut, "wt": wt, "hla": hla, "delta": delta, "scalars": scalars, "labels": labels}


def tensors_from_arrays(arrays: dict[str, np.ndarray], indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "mut_tokens": torch.as_tensor(arrays["mut"][indices], device=device),
        "wt_tokens": torch.as_tensor(arrays["wt"][indices], device=device),
        "hla_tokens": torch.as_tensor(arrays["hla"][indices], device=device),
        "delta_tokens": torch.as_tensor(arrays["delta"][indices], device=device),
        "scalars": torch.as_tensor(arrays["scalars"][indices], device=device),
        "labels": torch.as_tensor(arrays["labels"][indices], device=device),
    }


def run_training(mode: str, round_id: int, fold: int, checkpoint_name: str | None = None) -> dict[str, float]:
    settings = load_settings(ROOT)
    data_path = settings.data_processed / mode / "dataset.parquet"
    df = load_processed_dataset(data_path)
    train_df, val_df = split_frame(df, fold)
    cfg = TrainConfig()
    device = device_for_run()
    train_arrays = build_arrays(train_df)
    val_arrays = build_arrays(val_df)
    combined = torch.utils.data.TensorDataset(torch.arange(len(train_df)))
    loader = torch.utils.data.DataLoader(combined, batch_size=cfg.batch_size, shuffle=True)
    model = NeoantigenRanker(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    start = time.time()
    for _epoch in range(cfg.epochs):
        model.train()
        for (batch_indices,) in loader:
            batch = tensors_from_arrays(train_arrays, batch_indices.numpy(), device)
            logits = model(batch)
            loss = focal_bce(logits, batch["labels"], cfg.focal_gamma)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        val_scores = []
        val_labels = []
        for idx in range(len(val_df)):
            batch = tensors_from_arrays(val_arrays, np.array([idx]), device)
            val_scores.append(torch.sigmoid(model(batch)).item())
            val_labels.append(float(batch["labels"].item()))
    metrics = metric_bundle(np.asarray(val_labels), np.asarray(val_scores))
    metrics["training_seconds"] = time.time() - start
    metrics["peak_memory_mb"] = 0.0
    metrics["device"] = str(device)
    metrics["num_params"] = float(sum(p.numel() for p in model.parameters()))
    run_dir = settings.artifacts_runs / mode / f"round_{round_id:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_name or f"round_{round_id:02d}"
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, run_dir / f"{checkpoint}.pt")
    (run_dir / f"{checkpoint}.metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint-name", default=None)
    args = parser.parse_args()
    if args.mode == "full":
        raise RuntimeError("Full training is not enabled until full data ingest is completed.")
    metrics = run_training(args.mode, args.round_id, args.fold, args.checkpoint_name)
    print("---")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()

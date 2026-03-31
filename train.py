#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
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

# Search-space guidance:
# Keep edits focused on scalar feature usage, sequence encoders, WT-vs-Mut
# interaction structure, and the fusion block. Avoid micro-tuning the loss.
SCALAR_COLUMNS = [
    "mut_log_ba",
    "mut_el_score",
    "mut_ba_rank",
    "mut_el_rank",
    "wt_log_ba",
    "wt_el_score",
    "wt_ba_rank",
    "wt_el_rank",
    "stab_log_half_life",
    "stab_rank",
    "foreignness_score",
    "blast_bitscore",
    "blast_pident",
    "gravy",
    "aromaticity",
    "non_polar_ratio",
    "delta_fraction",
    "agretopicity",
    "peptide_length_scaled",
]


@dataclass
class TrainConfig:
    max_peptide_len: int = 11
    max_hla_len: int = 34
    peptide_embed_dim: int = 24
    hla_embed_dim: int = 24
    seq_hidden_dim: int = 32
    scalar_hidden_dim: int = 40
    fusion_hidden_dim: int = 72
    seq_kernel_size: int = 3
    dropout: float = 0.15
    lr: float = 8e-4
    weight_decay: float = 5e-4
    epochs: int = 55
    batch_size: int = 8
    seed: int = 7


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_for_run() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode_sequence(sequence: str, length: int) -> list[int]:
    ids = [AA_TO_ID.get(char, 0) for char in sequence[:length]]
    return ids + [0] * (length - len(ids))


def encode_delta_sequence(mut: str, wt: str, length: int) -> list[int]:
    tokens = [AA_TO_ID.get(mut_char, 0) if mut_char != wt_char else 0 for mut_char, wt_char in zip(mut, wt, strict=True)]
    tokens = tokens[:length]
    return tokens + [0] * (length - len(tokens))


def build_scalar_matrix(df) -> np.ndarray:
    return np.column_stack(
        [
            np.log1p(df["ba_score"].fillna(5e4).clip(lower=0.0)),
            df["el_score"].fillna(0.0),
            df["ba_rank"].fillna(100.0),
            df["el_rank"].fillna(100.0),
            np.log1p(df["wt_ba_score"].fillna(5e4).clip(lower=0.0)),
            df["wt_el_score"].fillna(0.0),
            df["wt_ba_rank"].fillna(100.0),
            df["wt_el_rank"].fillna(100.0),
            np.log1p(df["stab_score"].fillna(0.0).clip(lower=0.0)),
            df["stab_rank"].fillna(100.0),
            df["foreignness_score"].fillna(0.0),
            df["blast_bitscore"].fillna(0.0),
            df["blast_pident"].fillna(0.0),
            df["gravy"].fillna(0.0),
            df["aromaticity"].fillna(0.0),
            df["non_polar_ratio"].fillna(0.0),
            df["delta_fraction"].fillna(0.0),
            df["agretopicity"].fillna(0.0),
            df["peptide_length"].fillna(0.0) / 11.0,
        ]
    ).astype(np.float32)


def fit_scalar_stats(scalars: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = scalars.mean(axis=0, keepdims=True).astype(np.float32)
    std = scalars.std(axis=0, keepdims=True).astype(np.float32)
    std = np.clip(std, 0.05, None)
    return mean, std


def apply_scalar_stats(scalars: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((scalars - mean) / std).astype(np.float32)


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        max_len: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position = nn.Embedding(max_len, embed_dim)
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        mask = tokens != 0
        hidden = self.embedding(tokens) + self.position(positions)
        hidden = self.input_proj(hidden)
        convolved = self.conv(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = self.norm(hidden + self.dropout(convolved))

        masked_hidden = hidden * mask.unsqueeze(-1)
        mean_pool = masked_hidden.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)

        max_pool = hidden.masked_fill(~mask.unsqueeze(-1), -1e9).amax(dim=1)
        has_tokens = mask.any(dim=1, keepdim=True)
        max_pool = torch.where(has_tokens, max_pool, torch.zeros_like(max_pool))
        return torch.cat([mean_pool, max_pool], dim=1)


class ScalarTower(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, scalars: torch.Tensor) -> torch.Tensor:
        return self.net(self.input_norm(scalars))


class NeoantigenRanker(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        vocab_size = len(AA_TO_ID) + 1
        self.peptide_encoder = SequenceEncoder(
            vocab_size=vocab_size,
            embed_dim=cfg.peptide_embed_dim,
            hidden_dim=cfg.seq_hidden_dim,
            max_len=cfg.max_peptide_len,
            kernel_size=cfg.seq_kernel_size,
            dropout=cfg.dropout,
        )
        self.hla_encoder = SequenceEncoder(
            vocab_size=vocab_size,
            embed_dim=cfg.hla_embed_dim,
            hidden_dim=cfg.seq_hidden_dim,
            max_len=cfg.max_hla_len,
            kernel_size=cfg.seq_kernel_size,
            dropout=cfg.dropout,
        )
        self.scalar_tower = ScalarTower(len(SCALAR_COLUMNS), cfg.scalar_hidden_dim, cfg.dropout)

        seq_width = cfg.seq_hidden_dim * 2
        pair_width = seq_width * 7
        self.fusion_gate = nn.Sequential(
            nn.Linear(cfg.scalar_hidden_dim, pair_width),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(pair_width)
        self.head = nn.Sequential(
            nn.Linear(pair_width + cfg.scalar_hidden_dim, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden_dim, cfg.fusion_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden_dim // 2, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mut = self.peptide_encoder(batch["mut_tokens"])
        wt = self.peptide_encoder(batch["wt_tokens"])
        delta = self.peptide_encoder(batch["delta_tokens"])
        hla = self.hla_encoder(batch["hla_tokens"])
        scalars = self.scalar_tower(batch["scalars"])

        sequence_features = torch.cat(
            [
                mut,
                wt,
                delta,
                hla,
                mut - wt,
                mut * hla,
                delta * hla,
            ],
            dim=1,
        )
        gated_features = self.fusion_norm(sequence_features * self.fusion_gate(scalars))
        fused = torch.cat([gated_features, scalars], dim=1)
        return self.head(fused).squeeze(1)


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels)


def build_arrays(df, cfg: TrainConfig | None = None) -> dict[str, np.ndarray]:
    cfg = cfg or TrainConfig()
    mut = np.array([encode_sequence(item, cfg.max_peptide_len) for item in df["peptide_mut"]], dtype=np.int64)
    wt = np.array([encode_sequence(item, cfg.max_peptide_len) for item in df["peptide_wt"]], dtype=np.int64)
    hla = np.array([encode_sequence(item, cfg.max_hla_len) for item in df["hla_pseudosequence"]], dtype=np.int64)
    delta = np.array(
        [
            encode_delta_sequence(m, w, cfg.max_peptide_len)
            for m, w in zip(df["peptide_mut"], df["peptide_wt"], strict=True)
        ],
        dtype=np.int64,
    )
    scalars = build_scalar_matrix(df)
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
    cfg = TrainConfig()
    seed_everything(cfg.seed)
    settings = load_settings(ROOT)
    data_path = settings.data_processed / mode / "dataset.parquet"
    df = load_processed_dataset(data_path)
    train_df, val_df = split_frame(df, fold)
    if train_df.empty or val_df.empty:
        raise RuntimeError(f"Empty split encountered for fold={fold}: train={len(train_df)} val={len(val_df)}")
    device = device_for_run()
    train_arrays = build_arrays(train_df, cfg)
    val_arrays = build_arrays(val_df, cfg)
    scalar_mean, scalar_std = fit_scalar_stats(train_arrays["scalars"])
    train_arrays["scalars"] = apply_scalar_stats(train_arrays["scalars"], scalar_mean, scalar_std)
    val_arrays["scalars"] = apply_scalar_stats(val_arrays["scalars"], scalar_mean, scalar_std)

    indices = torch.arange(len(train_df), dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(indices),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    model = NeoantigenRanker(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    start = time.time()

    for _epoch in range(cfg.epochs):
        model.train()
        for (batch_indices,) in loader:
            batch = tensors_from_arrays(train_arrays, batch_indices.numpy(), device)
            logits = model(batch)
            loss = classification_loss(logits, batch["labels"])
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

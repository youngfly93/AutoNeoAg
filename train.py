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

from autoneoag.bootstrap import ensure_project_python

ensure_project_python(ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoneoag.config import load_settings
from autoneoag.dataset import load_processed_dataset, split_frame
from autoneoag.metrics.ranking import metric_bundle
from autoneoag.tasks import get_task_spec, list_task_ids, processed_dataset_path, run_dir as task_run_dir


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: idx + 1 for idx, aa in enumerate(AMINO_ACIDS)}

# Search-space guidance:
# Prefer higher-level changes over local pooling tweaks:
# 1. feature block composition and scalar comparison blocks
# 2. WT-vs-Mut contrast head structure
# 3. pair/group ranking objectives
# 4. only then sequence/fusion micro-architecture
FEATURE_BLOCK_COLUMNS = {
    "base": (
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
    ),
    "comparison": (
        "wt_minus_mut_log_ba",
        "mut_minus_wt_el",
        "wt_minus_mut_ba_rank",
        "wt_minus_mut_el_rank",
        "delta_x_agretopicity",
        "delta_x_foreignness",
    ),
    "context": (
        "delta_x_stability",
        "foreignness_x_stability",
        "mut_el_x_length",
        "blast_consistency",
    ),
}
DEFAULT_FEATURE_BLOCKS = ("base", "comparison", "context")
OBJECTIVE_REGISTRY = ("bce", "hybrid_pairwise", "pairwise_only")


@dataclass
class TrainConfig:
    max_peptide_len: int = 11
    max_hla_len: int = 34
    peptide_embed_dim: int = 24
    hla_embed_dim: int = 24
    seq_hidden_dim: int = 32
    scalar_hidden_dim: int = 48
    contrast_hidden_dim: int = 48
    fusion_hidden_dim: int = 96
    seq_kernel_size: int = 3
    dropout: float = 0.15
    lr: float = 8e-4
    weight_decay: float = 5e-4
    epochs: int = 55
    batch_size: int = 8
    seed: int = 7
    feature_blocks: tuple[str, ...] = DEFAULT_FEATURE_BLOCKS
    objective_mode: str = "bce"
    pairwise_weight: float = 0.25
    contrast_logit_weight: float = 0.30
    preference_logit_weight: float = 0.15


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_for_run() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def selected_scalar_columns(cfg: TrainConfig) -> list[str]:
    columns: list[str] = []
    for block_name in cfg.feature_blocks:
        if block_name not in FEATURE_BLOCK_COLUMNS:
            raise KeyError(f"Unknown feature block: {block_name}")
        columns.extend(FEATURE_BLOCK_COLUMNS[block_name])
    return columns


def scalar_input_dim(cfg: TrainConfig) -> int:
    return len(selected_scalar_columns(cfg))


def feature_block_slices(cfg: TrainConfig) -> dict[str, slice]:
    slices: dict[str, slice] = {}
    start = 0
    for block_name in cfg.feature_blocks:
        width = len(FEATURE_BLOCK_COLUMNS[block_name])
        slices[block_name] = slice(start, start + width)
        start += width
    return slices


def encode_sequence(sequence: str, length: int) -> list[int]:
    ids = [AA_TO_ID.get(char, 0) for char in sequence[:length]]
    return ids + [0] * (length - len(ids))


def encode_delta_sequence(mut: str, wt: str, length: int) -> list[int]:
    tokens = [AA_TO_ID.get(mut_char, 0) if mut_char != wt_char else 0 for mut_char, wt_char in zip(mut, wt, strict=True)]
    tokens = tokens[:length]
    return tokens + [0] * (length - len(tokens))


def build_feature_blocks(df) -> dict[str, np.ndarray]:
    mut_log_ba = np.log1p(df["ba_score"].fillna(5e4).clip(lower=0.0))
    wt_log_ba = np.log1p(df["wt_ba_score"].fillna(5e4).clip(lower=0.0))
    mut_el = df["el_score"].fillna(0.0)
    wt_el = df["wt_el_score"].fillna(0.0)
    mut_ba_rank = df["ba_rank"].fillna(100.0)
    wt_ba_rank = df["wt_ba_rank"].fillna(100.0)
    mut_el_rank = df["el_rank"].fillna(100.0)
    wt_el_rank = df["wt_el_rank"].fillna(100.0)
    stab_log_half_life = np.log1p(df["stab_score"].fillna(0.0).clip(lower=0.0))
    stab_rank = df["stab_rank"].fillna(100.0)
    foreignness = df["foreignness_score"].fillna(0.0)
    blast_bitscore = df["blast_bitscore"].fillna(0.0)
    blast_pident = df["blast_pident"].fillna(0.0)
    gravy = df["gravy"].fillna(0.0)
    aromaticity = df["aromaticity"].fillna(0.0)
    non_polar = df["non_polar_ratio"].fillna(0.0)
    delta_fraction = df["delta_fraction"].fillna(0.0)
    agretopicity = df["agretopicity"].fillna(0.0)
    peptide_length_scaled = df["peptide_length"].fillna(0.0) / 11.0

    base = np.column_stack(
        [
            mut_log_ba,
            mut_el,
            mut_ba_rank,
            mut_el_rank,
            wt_log_ba,
            wt_el,
            wt_ba_rank,
            wt_el_rank,
            stab_log_half_life,
            stab_rank,
            foreignness,
            blast_bitscore,
            blast_pident,
            gravy,
            aromaticity,
            non_polar,
            delta_fraction,
            agretopicity,
            peptide_length_scaled,
        ]
    ).astype(np.float32)

    comparison = np.column_stack(
        [
            wt_log_ba - mut_log_ba,
            mut_el - wt_el,
            wt_ba_rank - mut_ba_rank,
            wt_el_rank - mut_el_rank,
            delta_fraction * agretopicity,
            delta_fraction * foreignness,
        ]
    ).astype(np.float32)

    context = np.column_stack(
        [
            delta_fraction * stab_log_half_life,
            foreignness * stab_log_half_life,
            mut_el * peptide_length_scaled,
            blast_bitscore * (blast_pident / 100.0),
        ]
    ).astype(np.float32)

    return {
        "base": base,
        "comparison": comparison,
        "context": context,
    }


def build_scalar_matrix(df, cfg: TrainConfig) -> np.ndarray:
    blocks = build_feature_blocks(df)
    matrices = [blocks[block_name] for block_name in cfg.feature_blocks]
    return np.concatenate(matrices, axis=1).astype(np.float32)


def encode_group_ids(df) -> np.ndarray:
    lookup: dict[str, int] = {}
    encoded = []
    for study_id, hla in zip(df["study_id"], df["hla"], strict=True):
        key = f"{study_id}|{hla}"
        if key not in lookup:
            lookup[key] = len(lookup)
        encoded.append(lookup[key])
    return np.asarray(encoded, dtype=np.int64)


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


class ContrastHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.logit = nn.Linear(hidden_dim, 1)

    def forward(self, contrast_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(contrast_inputs)
        return hidden, self.logit(hidden).squeeze(1)


class AllelePreferenceHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.logit = nn.Linear(hidden_dim, 1)

    def forward(self, pair_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(pair_inputs)
        return hidden, self.logit(hidden).squeeze(1)


class ContrastConditioner(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

    def forward(self, scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift = self.net(scalars).chunk(2, dim=1)
        return scale, shift


class FusionBlock(nn.Module):
    def __init__(self, seq_dim: int, conditioning_dim: int, dropout: float) -> None:
        super().__init__()
        self.affine = nn.Linear(conditioning_dim, seq_dim * 2)
        self.residual = nn.Linear(conditioning_dim, seq_dim)
        self.norm = nn.LayerNorm(seq_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence_features: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale, shift = self.affine(conditioning).chunk(2, dim=1)
        modulated = sequence_features * (1.0 + 0.1 * torch.tanh(scale))
        modulated = modulated + 0.1 * shift + 0.1 * self.residual(conditioning)
        return self.norm(sequence_features + self.dropout(modulated))


class NeoantigenRanker(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.block_slices = feature_block_slices(cfg)
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
        self.scalar_tower = ScalarTower(scalar_input_dim(cfg), cfg.scalar_hidden_dim, cfg.dropout)

        seq_width = cfg.seq_hidden_dim * 2
        preference_input_dim = seq_width * 4
        self.preference_head = AllelePreferenceHead(preference_input_dim, cfg.contrast_hidden_dim, cfg.dropout)
        contrast_input_dim = seq_width * 7 + cfg.contrast_hidden_dim
        self.contrast_head = ContrastHead(contrast_input_dim, cfg.contrast_hidden_dim, cfg.dropout)
        contrast_scalar_dim = sum(
            len(FEATURE_BLOCK_COLUMNS[block_name])
            for block_name in cfg.feature_blocks
            if block_name in {"comparison", "context"}
        )
        self.contrast_conditioner = (
            ContrastConditioner(contrast_scalar_dim, cfg.contrast_hidden_dim, cfg.dropout)
            if contrast_scalar_dim > 0
            else None
        )

        pair_width = seq_width * 8
        conditioning_dim = cfg.scalar_hidden_dim + cfg.contrast_hidden_dim
        self.fusion = FusionBlock(pair_width, conditioning_dim, cfg.dropout)
        self.head = nn.Sequential(
            nn.Linear(pair_width + conditioning_dim, cfg.fusion_hidden_dim),
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
        contrast_scalar_parts = []
        for block_name in ("comparison", "context"):
            block_slice = self.block_slices.get(block_name)
            if block_slice is not None:
                contrast_scalar_parts.append(batch["scalars"][:, block_slice])

        average_peptide = 0.5 * (mut + wt)
        mut_hla = mut * hla
        wt_hla = wt * hla
        mut_preference_inputs = torch.cat([mut, hla, mut_hla, torch.abs(mut - hla)], dim=1)
        wt_preference_inputs = torch.cat([wt, hla, wt_hla, torch.abs(wt - hla)], dim=1)
        mut_preference_hidden, mut_preference_logit = self.preference_head(mut_preference_inputs)
        wt_preference_hidden, wt_preference_logit = self.preference_head(wt_preference_inputs)
        preference_delta_hidden = mut_preference_hidden - wt_preference_hidden
        preference_delta_logit = mut_preference_logit - wt_preference_logit
        contrast_inputs = torch.cat(
            [
                mut - wt,
                torch.abs(mut - wt),
                delta,
                mut * wt,
                mut_hla - wt_hla,
                (mut - wt) * hla,
                delta * hla,
                preference_delta_hidden,
            ],
            dim=1,
        )
        contrast_hidden, contrast_logit = self.contrast_head(contrast_inputs)
        if self.contrast_conditioner is not None and contrast_scalar_parts:
            contrast_scalar_inputs = torch.cat(contrast_scalar_parts, dim=1)
            contrast_scale, contrast_shift = self.contrast_conditioner(contrast_scalar_inputs)
            contrast_hidden = contrast_hidden * (1.0 + 0.1 * torch.tanh(contrast_scale))
            contrast_hidden = contrast_hidden + 0.1 * contrast_shift

        sequence_features = torch.cat(
            [
                mut,
                wt,
                delta,
                hla,
                mut * wt,
                average_peptide * hla,
                mut - wt,
                (mut - wt) * hla,
            ],
            dim=1,
        )
        conditioning = torch.cat([scalars, contrast_hidden], dim=1)
        fused = self.fusion(sequence_features, conditioning)
        main_logit = self.head(torch.cat([fused, conditioning], dim=1)).squeeze(1)
        return (
            main_logit
            + self.cfg.contrast_logit_weight * contrast_logit
            + self.cfg.preference_logit_weight * preference_delta_logit
        )


def pairwise_ranking_loss(logits: torch.Tensor, labels: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    losses = []
    for group_id in torch.unique(group_ids):
        mask = group_ids == group_id
        if int(mask.sum().item()) < 2:
            continue
        group_logits = logits[mask]
        group_labels = labels[mask]
        positives = group_logits[group_labels > 0.5]
        negatives = group_logits[group_labels <= 0.5]
        if positives.numel() == 0 or negatives.numel() == 0:
            continue
        losses.append(F.softplus(-(positives.unsqueeze(1) - negatives.unsqueeze(0))).mean())
    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


def objective_loss(logits: torch.Tensor, labels: torch.Tensor, group_ids: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    if cfg.objective_mode not in OBJECTIVE_REGISTRY:
        raise KeyError(f"Unknown objective mode: {cfg.objective_mode}")
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    pairwise = pairwise_ranking_loss(logits, labels, group_ids)
    if cfg.objective_mode == "bce":
        return bce
    if cfg.objective_mode == "hybrid_pairwise":
        return bce + cfg.pairwise_weight * pairwise
    return pairwise


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
    scalars = build_scalar_matrix(df, cfg)
    labels = df["label"].to_numpy(dtype=np.float32)
    group_ids = encode_group_ids(df)
    return {
        "mut": mut,
        "wt": wt,
        "hla": hla,
        "delta": delta,
        "scalars": scalars,
        "labels": labels,
        "group_ids": group_ids,
    }


def tensors_from_arrays(arrays: dict[str, np.ndarray], indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "mut_tokens": torch.as_tensor(arrays["mut"][indices], device=device),
        "wt_tokens": torch.as_tensor(arrays["wt"][indices], device=device),
        "hla_tokens": torch.as_tensor(arrays["hla"][indices], device=device),
        "delta_tokens": torch.as_tensor(arrays["delta"][indices], device=device),
        "scalars": torch.as_tensor(arrays["scalars"][indices], device=device),
        "labels": torch.as_tensor(arrays["labels"][indices], device=device),
        "group_ids": torch.as_tensor(arrays["group_ids"][indices], device=device),
    }


def dev_folds(df) -> list[int]:
    folds = sorted({int(fold) for fold in df.loc[df["split"] == "dev", "fold"].tolist() if int(fold) >= 0})
    if not folds:
        raise RuntimeError("No development folds are available for training.")
    return folds


def fit_model(
    train_df,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[NeoantigenRanker, np.ndarray, np.ndarray]:
    train_arrays = build_arrays(train_df, cfg)
    scalar_mean, scalar_std = fit_scalar_stats(train_arrays["scalars"])
    train_arrays["scalars"] = apply_scalar_stats(train_arrays["scalars"], scalar_mean, scalar_std)

    effective_batch_size = len(train_df) if cfg.objective_mode != "bce" else min(cfg.batch_size, len(train_df))
    shuffle = effective_batch_size < len(train_df)
    indices = torch.arange(len(train_df), dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(indices),
        batch_size=max(1, effective_batch_size),
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    model = NeoantigenRanker(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for _epoch in range(cfg.epochs):
        model.train()
        for (batch_indices,) in loader:
            batch = tensors_from_arrays(train_arrays, batch_indices.numpy(), device)
            logits = model(batch)
            loss = objective_loss(logits, batch["labels"], batch["group_ids"], cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model, scalar_mean, scalar_std


def predict_scores(
    model: NeoantigenRanker,
    frame,
    cfg: TrainConfig,
    device: torch.device,
    scalar_mean: np.ndarray,
    scalar_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = build_arrays(frame, cfg)
    arrays["scalars"] = apply_scalar_stats(arrays["scalars"], scalar_mean, scalar_std)
    scores, labels = [], []
    model.eval()
    with torch.no_grad():
        for idx in range(len(frame)):
            batch = tensors_from_arrays(arrays, np.array([idx]), device)
            scores.append(torch.sigmoid(model(batch)).item())
            labels.append(float(batch["labels"].item()))
    return np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def save_checkpoint(
    path: Path,
    model: NeoantigenRanker,
    cfg: TrainConfig,
    scalar_mean: np.ndarray,
    scalar_std: np.ndarray,
    metadata: dict[str, object] | None = None,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(cfg),
        "scalar_mean": scalar_mean.tolist(),
        "scalar_std": scalar_std.tolist(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def run_training(
    task_id: str,
    mode: str,
    strategy: str,
    run_id: int,
    round_id: int,
    fold: int | None,
    checkpoint_name: str | None = None,
) -> dict[str, float | int | str]:
    cfg = TrainConfig()
    seed_everything(cfg.seed)
    settings = load_settings(ROOT)
    task = get_task_spec(task_id)
    data_path = processed_dataset_path(settings, task.task_id, mode)
    df = load_processed_dataset(data_path)
    device = device_for_run()
    start = time.time()
    run_dir = task_run_dir(settings, task.task_id, mode, strategy, run_id, round_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_name or f"round_{round_id:02d}"
    checkpoint_path = run_dir / f"{checkpoint}.pt"

    if fold is None:
        folds = dev_folds(df)
        oof_scores = []
        oof_labels = []
        fold_metrics: dict[str, dict[str, float]] = {}
        for fold_id in folds:
            train_df, val_df = split_frame(df, fold_id)
            if train_df.empty or val_df.empty:
                raise RuntimeError(f"Empty split encountered for fold={fold_id}: train={len(train_df)} val={len(val_df)}")
            model, scalar_mean, scalar_std = fit_model(train_df, cfg, device)
            fold_scores, fold_labels = predict_scores(model, val_df, cfg, device, scalar_mean, scalar_std)
            fold_metric = metric_bundle(fold_labels, fold_scores)
            fold_metrics[str(fold_id)] = fold_metric
            oof_scores.append(fold_scores)
            oof_labels.append(fold_labels)
        full_dev = df[df["split"] == "dev"].reset_index(drop=True)
        final_model, final_scalar_mean, final_scalar_std = fit_model(full_dev, cfg, device)
        all_scores = np.concatenate(oof_scores)
        all_labels = np.concatenate(oof_labels)
        metrics = metric_bundle(all_labels, all_scores)
        metrics["cv_num_folds"] = len(folds)
        metrics["cv_num_samples"] = int(len(all_labels))
        metrics["selection_mode"] = "grouped_cv_oof"
        metrics_payload = {
            **metrics,
            "fold_metrics": fold_metrics,
            "feature_blocks": list(cfg.feature_blocks),
            "objective_mode": cfg.objective_mode,
            "selected_scalar_columns": selected_scalar_columns(cfg),
        }
        save_checkpoint(
            checkpoint_path,
            final_model,
            cfg,
            final_scalar_mean,
            final_scalar_std,
            metadata={
                "selection_mode": "grouped_cv_oof",
                "folds": folds,
                "train_split": "dev",
                "feature_blocks": list(cfg.feature_blocks),
                "objective_mode": cfg.objective_mode,
            },
        )
        num_params = float(sum(p.numel() for p in final_model.parameters()))
    else:
        train_df, val_df = split_frame(df, fold)
        if train_df.empty or val_df.empty:
            raise RuntimeError(f"Empty split encountered for fold={fold}: train={len(train_df)} val={len(val_df)}")
        model, scalar_mean, scalar_std = fit_model(train_df, cfg, device)
        val_scores, val_labels = predict_scores(model, val_df, cfg, device, scalar_mean, scalar_std)
        metrics = metric_bundle(val_labels, val_scores)
        metrics["cv_num_folds"] = 1
        metrics["cv_num_samples"] = int(len(val_labels))
        metrics["selection_mode"] = "single_fold"
        metrics_payload = {
            **metrics,
            "feature_blocks": list(cfg.feature_blocks),
            "objective_mode": cfg.objective_mode,
            "selected_scalar_columns": selected_scalar_columns(cfg),
        }
        save_checkpoint(
            checkpoint_path,
            model,
            cfg,
            scalar_mean,
            scalar_std,
            metadata={
                "selection_mode": "single_fold",
                "fold": fold,
                "train_split": "dev",
                "feature_blocks": list(cfg.feature_blocks),
                "objective_mode": cfg.objective_mode,
            },
        )
        num_params = float(sum(p.numel() for p in model.parameters()))

    metrics["training_seconds"] = time.time() - start
    metrics["peak_memory_mb"] = 0.0
    metrics["device"] = str(device)
    metrics["num_params"] = num_params
    metrics["task_id"] = task.task_id
    metrics["strategy"] = strategy
    metrics["run_id"] = run_id
    metrics["round_id"] = round_id
    metrics_payload["training_seconds"] = metrics["training_seconds"]
    metrics_payload["peak_memory_mb"] = metrics["peak_memory_mb"]
    metrics_payload["device"] = metrics["device"]
    metrics_payload["num_params"] = metrics["num_params"]
    metrics_payload["task_id"] = metrics["task_id"]
    metrics_payload["strategy"] = metrics["strategy"]
    metrics_payload["run_id"] = metrics["run_id"]
    metrics_payload["round_id"] = metrics["round_id"]
    (run_dir / f"{checkpoint}.metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list_task_ids(), required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--strategy", choices=["constrained", "random", "unconstrained"], default="constrained")
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--checkpoint-name", default=None)
    args = parser.parse_args()
    if args.mode == "full":
        raise RuntimeError("Full training is not enabled until full data ingest is completed.")
    metrics = run_training(args.task, args.mode, args.strategy, args.run_id, args.round_id, args.fold, args.checkpoint_name)
    print("---")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()

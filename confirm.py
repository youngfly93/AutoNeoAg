#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

import train as train_mod
from autoneoag.config import load_settings
from autoneoag.dataset import load_processed_dataset
from autoneoag.metrics.ranking import metric_bundle


def evaluate_split(mode: str, checkpoint: str, split_name: str) -> dict[str, float]:
    settings = load_settings(ROOT)
    df = load_processed_dataset(settings.data_processed / mode / "dataset.parquet")
    split_df = df[df["split"] == split_name].reset_index(drop=True)
    model_payload = torch.load(Path(checkpoint), map_location="cpu")
    cfg = train_mod.TrainConfig(**model_payload["config"])
    model = train_mod.NeoantigenRanker(cfg)
    model.load_state_dict(model_payload["state_dict"])
    model.eval()
    arrays = train_mod.build_arrays(split_df, cfg)
    scores, labels = [], []
    with torch.no_grad():
        for idx in range(len(split_df)):
            batch = train_mod.tensors_from_arrays(arrays, np.array([idx]), torch.device("cpu"))
            scores.append(torch.sigmoid(model(batch)).item())
            labels.append(float(batch["labels"].item()))
    metrics = metric_bundle(np.asarray(labels), np.asarray(scores))
    metrics["split"] = split_name
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    if args.mode == "full":
        raise RuntimeError("Full confirm requires completed full ingest.")
    metrics = evaluate_split(args.mode, args.checkpoint, "confirm")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

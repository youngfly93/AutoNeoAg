#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from autoneoag.bootstrap import ensure_project_python

ensure_project_python(ROOT)

import numpy as np
import torch

import train as train_mod
from autoneoag.config import load_settings
from autoneoag.dataset import load_processed_dataset
from autoneoag.metrics.ranking import metric_bundle
from autoneoag.tasks import get_task_spec, list_task_ids, processed_dataset_path


def evaluate_split(task_id: str, mode: str, checkpoint: str, split_name: str) -> dict[str, float]:
    settings = load_settings(ROOT)
    task = get_task_spec(task_id)
    dataset_path = processed_dataset_path(settings, task.task_id, mode)
    if not dataset_path.exists():
        raise RuntimeError(
            f"Processed dataset is missing for task={task.task_id} mode={mode}. "
            f"Run `python prepare.py --task {task.task_id} --mode {mode}` first."
        )
    df = load_processed_dataset(dataset_path)
    split_df = df[df["split"] == split_name].reset_index(drop=True)
    if split_df.empty:
        raise RuntimeError(f"Split {split_name!r} is empty for task={task.task_id} mode={mode}.")
    model_payload = torch.load(Path(checkpoint), map_location="cpu")
    cfg = train_mod.TrainConfig(**model_payload["config"])
    model = train_mod.NeoantigenRanker(cfg)
    model.load_state_dict(model_payload["state_dict"])
    model.eval()
    arrays = train_mod.build_arrays(split_df, cfg)
    scalar_mean = model_payload.get("scalar_mean")
    scalar_std = model_payload.get("scalar_std")
    if scalar_mean is not None and scalar_std is not None:
        arrays["scalars"] = train_mod.apply_scalar_stats(
            arrays["scalars"],
            np.asarray(scalar_mean, dtype=np.float32),
            np.asarray(scalar_std, dtype=np.float32),
        )
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
    parser.add_argument("--task", choices=list_task_ids(), required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    metrics = evaluate_split(args.task, args.mode, args.checkpoint, "confirm")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

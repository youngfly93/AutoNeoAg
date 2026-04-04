#!/Volumes/AutoNeoAgEnv/autoneoag-py312/bin/python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _strategy_label(task_id: str, strategy: str) -> str:
    return f"{task_id}\n{strategy}"


def plot_trust_summary(summary_tsv: Path, outpath: Path) -> Path:
    df = pd.read_csv(summary_tsv, sep="\t")
    if df.empty:
        raise RuntimeError(f"No rows found in {summary_tsv}")

    grouped = (
        df.groupby(["task_id", "strategy"], as_index=False)
        .agg(
            keep_rate=("keep_rate", "mean"),
            confirm_gate_pass_rate=("confirm_gate_pass_rate", "mean"),
            winner_changed_rate=("winner_changed_by_strict_confirm", "mean"),
            best_keep_dev=("best_keep_dev", "mean"),
            report_confirm=("report_confirm", "mean"),
            report_blind=("report_blind", "mean"),
        )
        .sort_values(["task_id", "strategy"])
        .reset_index(drop=True)
    )
    grouped["label"] = [_strategy_label(task, strategy) for task, strategy in zip(grouped["task_id"], grouped["strategy"])]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    x = range(len(grouped))
    width = 0.24

    axes[0, 0].bar([i - width for i in x], grouped["best_keep_dev"], width=width, label="best keep dev", color="#2f80ed")
    axes[0, 0].bar(x, grouped["report_confirm"], width=width, label="final confirm", color="#27ae60")
    axes[0, 0].bar([i + width for i in x], grouped["report_blind"], width=width, label="final blind", color="#d35400")
    axes[0, 0].set_title("dev-confirm-blind")
    axes[0, 0].set_xticks(list(x))
    axes[0, 0].set_xticklabels(grouped["label"])
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].legend()
    axes[0, 0].grid(True, axis="y", alpha=0.2)

    axes[0, 1].bar(x, grouped["keep_rate"], color="#8e44ad")
    axes[0, 1].set_title("keep rate")
    axes[0, 1].set_xticks(list(x))
    axes[0, 1].set_xticklabels(grouped["label"])
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].grid(True, axis="y", alpha=0.2)

    axes[1, 0].bar(x, grouped["confirm_gate_pass_rate"], color="#16a085")
    axes[1, 0].set_title("confirm gate pass rate")
    axes[1, 0].set_xticks(list(x))
    axes[1, 0].set_xticklabels(grouped["label"])
    axes[1, 0].set_ylim(0, 1.0)
    axes[1, 0].grid(True, axis="y", alpha=0.2)

    axes[1, 1].bar(x, grouped["winner_changed_rate"], color="#c0392b")
    axes[1, 1].set_title("winner changed by strict-confirm")
    axes[1, 1].set_xticks(list(x))
    axes[1, 1].set_xticklabels(grouped["label"])
    axes[1, 1].set_ylim(0, 1.0)
    axes[1, 1].grid(True, axis="y", alpha=0.2)

    fig.suptitle("Strict-confirm trust-oriented summary", fontsize=16)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "figures" / "strict_confirm_trust_summary.png")
    args = parser.parse_args()
    out = plot_trust_summary(args.summary.expanduser().resolve(), args.out.expanduser().resolve())
    print(out)


if __name__ == "__main__":
    main()

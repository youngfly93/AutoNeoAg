#!/Volumes/AutoNeoAgEnv/autoneoag-py312/bin/python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results.tsv"
DEFAULT_OUTDIR = ROOT / "artifacts" / "figures"


def _shorten(text: str, limit: int = 52) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    for col in ("run_id", "round_id", "lines_changed"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("dev_score", "confirm_score", "blind_score", "training_seconds"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def latest_run_slice(df: pd.DataFrame, task_id: str, strategy: str) -> pd.DataFrame:
    subset = df[(df["task_id"] == task_id) & (df["strategy"] == strategy)].copy()
    if subset.empty:
        return subset
    latest_run = int(subset["run_id"].max())
    subset = subset[subset["run_id"] == latest_run].copy()
    subset = subset.sort_values("round_id").reset_index(drop=True)
    subset["exp_idx"] = range(1, len(subset) + 1)
    return subset


def plot_task_progress(df: pd.DataFrame, task_id: str, strategy: str, outpath: Path) -> None:
    subset = latest_run_slice(df, task_id, strategy)
    if subset.empty:
        raise RuntimeError(f"No rows found for task={task_id} strategy={strategy}")

    valid = subset[subset["dev_score"].notna()].copy()
    if valid.empty:
        raise RuntimeError(f"No valid dev_score rows for task={task_id} strategy={strategy}")

    baseline = float(valid.iloc[0]["dev_score"])
    best = float(valid["dev_score"].max())
    keep_mask = valid["status"].str.lower() == "keep"
    discard_mask = valid["status"].str.lower() == "discard"

    fig, ax = plt.subplots(figsize=(16, 8))

    disc = valid[discard_mask]
    ax.scatter(
        disc["exp_idx"],
        disc["dev_score"],
        c="#c8ccd1",
        s=20,
        alpha=0.60,
        zorder=2,
        label="Discarded",
    )

    kept = valid[keep_mask].copy()
    ax.scatter(
        kept["exp_idx"],
        kept["dev_score"],
        c="#2ecc71",
        s=58,
        edgecolors="black",
        linewidths=0.5,
        zorder=4,
        label="Kept",
    )

    running_best = valid["dev_score"].cummax()
    ax.step(
        valid["exp_idx"],
        running_best,
        where="post",
        color="#27ae60",
        linewidth=2.2,
        alpha=0.85,
        zorder=3,
        label="Running best",
    )

    ax.axhline(baseline, color="#7f8c8d", linestyle="--", linewidth=1.2, alpha=0.6, label="Baseline")

    for _, row in kept.iterrows():
        label = f"r{int(row['round_id'])}: {_shorten(row['description'])}"
        ax.annotate(
            label,
            (row["exp_idx"], row["dev_score"]),
            textcoords="offset points",
            xytext=(6, 7),
            fontsize=8,
            color="#176b34",
            rotation=28,
            ha="left",
            va="bottom",
            alpha=0.92,
        )

    n_total = len(subset)
    n_keep = int((subset["status"].str.lower() == "keep").sum())
    latest_round = int(subset["round_id"].max())
    keep_rate = n_keep / n_total if n_total else 0.0
    improvement = best - baseline

    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel("Validation Score (higher is better)", fontsize=12)
    ax.set_title(
        f"{task_id} autoresearch progress: {n_total} experiments, {n_keep} kept, "
        f"best {best:.6f} (+{improvement:.6f} vs baseline)",
        fontsize=14,
    )
    ax.grid(True, alpha=0.18)
    ax.legend(loc="lower right", fontsize=9)

    y_span = max(best - valid["dev_score"].min(), 0.01)
    ax.set_ylim(valid["dev_score"].min() - 0.12 * y_span, best + 0.18 * y_span)
    ax.text(
        0.01,
        0.98,
        f"strategy={strategy}\nlatest_round={latest_round}\nkeep_rate={keep_rate:.1%}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.85},
    )

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_combined(df: pd.DataFrame, task_ids: list[str], strategy: str, outpath: Path) -> None:
    fig, axes = plt.subplots(len(task_ids), 1, figsize=(16, 7 * len(task_ids)), squeeze=False)
    axes = axes[:, 0]
    for ax, task_id in zip(axes, task_ids):
        subset = latest_run_slice(df, task_id, strategy)
        if subset.empty:
            ax.set_axis_off()
            continue
        valid = subset[subset["dev_score"].notna()].copy()
        baseline = float(valid.iloc[0]["dev_score"])
        best = float(valid["dev_score"].max())
        kept = valid[valid["status"].str.lower() == "keep"].copy()
        disc = valid[valid["status"].str.lower() == "discard"].copy()
        ax.scatter(disc["exp_idx"], disc["dev_score"], c="#c8ccd1", s=18, alpha=0.55, zorder=2)
        ax.scatter(kept["exp_idx"], kept["dev_score"], c="#2ecc71", s=48, edgecolors="black", linewidths=0.4, zorder=4)
        ax.step(valid["exp_idx"], valid["dev_score"].cummax(), where="post", color="#27ae60", linewidth=2.0, zorder=3)
        ax.axhline(baseline, color="#7f8c8d", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_title(
            f"{task_id}: best {best:.6f}, baseline {baseline:.6f}, "
            f"+{best - baseline:.6f}, keeps {len(kept)}/{len(subset)}",
            fontsize=13,
        )
        ax.set_xlabel("Experiment #")
        ax.set_ylabel("Validation Score")
        ax.grid(True, alpha=0.18)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--strategy", default="constrained")
    parser.add_argument("--tasks", nargs="+", default=["neoantigen", "hla_immunogenicity"])
    args = parser.parse_args()

    df = load_results(args.results)
    for task_id in args.tasks:
        plot_task_progress(df, task_id, args.strategy, args.outdir / f"{task_id}_progress.png")
    if len(args.tasks) > 1:
        plot_combined(df, args.tasks, args.strategy, args.outdir / "autoresearch_progress_combined.png")


if __name__ == "__main__":
    main()

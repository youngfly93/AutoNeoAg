#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from confirm import evaluate_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    if args.mode == "full":
        raise RuntimeError("Full blind evaluation requires completed full ingest and credentials.")
    metrics = evaluate_split(args.mode, args.checkpoint, "blind")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


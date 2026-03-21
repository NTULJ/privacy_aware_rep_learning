from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_linear_probe as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Privacy linear probe (verification).")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--feature-key", required=True)
    parser.add_argument("--pair-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--metric", choices=("cosine", "linear_probe"), default="linear_probe")
    parser.add_argument("--far-target", type=float, default=0.01)

    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--id-a-key", default="sample_id_a")
    parser.add_argument("--id-b-key", default="sample_id_b")
    parser.add_argument("--pair-label-key", default="pair_label")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--class-weight", choices=("none", "balanced"), default="none")
    parser.add_argument("--task", default="verification")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    features, index_rows = base.load_features(args.feature_root, args.feature_key)
    base.run_verification(features, index_rows, args)


if __name__ == "__main__":
    main()

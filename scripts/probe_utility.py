from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_linear_probe as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Utility linear probe (classification) with reusable head checkpoints.")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--feature-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--method", default="ours")
    parser.add_argument("--stage", default="hpool")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--model-dir", type=Path, default=Path("models/probes"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--force-retrain", action="store_true")

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
    parser.add_argument("--class-weight", choices=("none", "balanced"), default="none")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def sanitize_feature_key(feature_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", feature_key).strip("._") or "feature"


def resolve_model_path(args: argparse.Namespace) -> Path:
    if args.model_path is not None:
        return args.model_path
    feature_key_slug = sanitize_feature_key(args.feature_key)
    filename = f"{args.dataset_name}_{args.method}_{args.stage}_{feature_key_slug}_utility.pt"
    return args.model_dir / filename


def write_reuse_outputs(
    args: argparse.Namespace,
    index_rows: list[dict[str, str]],
    labels: list[str],
    splits: base.SplitIndices,
    y: np.ndarray,
    class_names: list[str],
    model: nn.Module,
    features: np.ndarray,
    model_payload: dict[str, object],
) -> None:
    device = torch.device(args.device)
    train_x = features[splits.train]
    val_x = features[splits.val] if splits.val.size else np.zeros((0, features.shape[1]), dtype=np.float32)
    test_x = features[splits.test]

    train_y = y[splits.train]
    val_y = y[splits.val] if splits.val.size else np.zeros((0,), dtype=np.int64)
    test_y = y[splits.test]

    train_metrics = base.evaluate_classifier(model, train_x, train_y, device)
    val_metrics = base.evaluate_classifier(model, val_x, val_y, device) if splits.val.size else train_metrics
    test_metrics = base.evaluate_classifier(model, test_x, test_y, device)

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "confusion_matrix.npy", test_metrics["confusion"])

    with (args.output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "task": "classification",
                "feature_key": args.feature_key,
                "reused_model": True,
                "model_path": str(resolve_model_path(args)),
                "best_epoch": int(model_payload.get("best_epoch", 0)),
                "train": {
                    "accuracy": train_metrics["accuracy"],
                    "macro_f1": train_metrics["macro_f1"],
                    "num_samples": int(splits.train.size),
                },
                "val": {
                    "accuracy": val_metrics["accuracy"],
                    "macro_f1": val_metrics["macro_f1"],
                    "num_samples": int(splits.val.size),
                },
                "test": {
                    "accuracy": test_metrics["accuracy"],
                    "macro_f1": test_metrics["macro_f1"],
                    "num_samples": int(splits.test.size),
                },
            },
            handle,
            indent=2,
        )

    with (args.output / "train_log.json").open("w", encoding="utf-8") as handle:
        json.dump([], handle, indent=2)

    with (args.output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "label", "pred_label"])
        writer.writeheader()
        test_pred = test_metrics["pred"]
        for idx, pred in zip(splits.test.tolist(), test_pred.tolist()):
            writer.writerow(
                {
                    "sample_id": index_rows[idx]["sample_id"],
                    "split": index_rows[idx].get("split", "all"),
                    "label": labels[idx],
                    "pred_label": class_names[pred],
                }
            )

    torch.save(model_payload, args.output / "probe_model.pt")
    print(f"reused utility head and saved outputs to {args.output}")


def run_with_reused_model(args: argparse.Namespace, features: np.ndarray, index_rows: list[dict[str, str]], model_path: Path) -> None:
    payload = torch.load(model_path, map_location="cpu")
    state_dict = payload.get("state_dict")
    class_names = list(payload.get("class_names", []))
    class_to_idx = dict(payload.get("class_to_idx", {}))

    if not state_dict or not class_names or not class_to_idx:
        raise ValueError(f"Invalid utility head checkpoint: {model_path}")

    checkpoint_feature_key = payload.get("feature_key")
    if isinstance(checkpoint_feature_key, str) and checkpoint_feature_key != args.feature_key:
        raise ValueError(
            f"Utility head feature_key mismatch: checkpoint={checkpoint_feature_key}, requested={args.feature_key}"
        )

    labels = [row.get(args.label_key, row.get("label", "")) for row in index_rows]
    unknown_labels = sorted({label for label in labels if label not in class_to_idx})
    if unknown_labels:
        preview = ", ".join(unknown_labels[:5])
        raise ValueError(f"Found labels not present in reused checkpoint: {preview}")

    y = np.asarray([int(class_to_idx[label]) for label in labels], dtype=np.int64)
    splits = base.resolve_splits(index_rows, labels, args)

    mean = payload.get("mean")
    std = payload.get("std")
    if isinstance(mean, np.ndarray) and isinstance(std, np.ndarray):
        features = base.apply_standardizer(features, mean.astype(np.float32), std.astype(np.float32))

    device = torch.device(args.device)
    model = nn.Linear(features.shape[1], len(class_names)).to(device)
    model.load_state_dict(state_dict)

    write_reuse_outputs(
        args=args,
        index_rows=index_rows,
        labels=labels,
        splits=splits,
        y=y,
        class_names=class_names,
        model=model,
        features=features,
        model_payload=payload,
    )


def run_training(args: argparse.Namespace, features: np.ndarray, index_rows: list[dict[str, str]], model_path: Path) -> None:
    base.train_classification_probe(features, index_rows, args)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.output / "probe_model.pt", model_path)

    metrics_path = args.output / "metrics.json"
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["reused_model"] = False
        payload["model_path"] = str(model_path)
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"trained utility head saved to {model_path}")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = resolve_model_path(args)

    features, index_rows = base.load_features(args.feature_root, args.feature_key)
    if model_path.exists() and not args.force_retrain:
        run_with_reused_model(args, features, index_rows, model_path)
        return

    run_training(args, features, index_rows, model_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight probes on frozen Qwen3-VL feature exports.")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--feature-key", required=True, help="Feature key inside pooled/, e.g. hpool_priv__global_mean")
    parser.add_argument("--task", choices=("classification", "verification"), required=True)
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--output", type=Path, required=True)
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
    parser.add_argument("--pair-file", type=Path, default=None)
    parser.add_argument("--id-a-key", default="sample_id_a")
    parser.add_argument("--id-b-key", default="sample_id_b")
    parser.add_argument("--pair-label-key", default="pair_label")
    parser.add_argument("--metric", choices=("cosine", "linear_probe"), default="cosine")
    parser.add_argument("--far-target", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_features(feature_root: Path, feature_key: str) -> tuple[np.ndarray, list[dict[str, str]]]:
    feature_path = feature_root / "pooled" / f"{feature_key}.npy"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature file: {feature_path}")
    features = np.load(feature_path).astype(np.float32)
    index_rows = read_csv_rows(feature_root / "index.csv")
    if features.shape[0] != len(index_rows):
        raise ValueError(f"Feature count {features.shape[0]} does not match index rows {len(index_rows)}")
    return features, index_rows


def stratified_random_split(labels: list[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> SplitIndices:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")
    rng = np.random.default_rng(seed)
    label_to_indices: dict[str, list[int]] = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for indices in label_to_indices.values():
        arr = np.asarray(indices, dtype=np.int64)
        rng.shuffle(arr)
        n = len(arr)
        n_train = max(1, int(round(n * train_ratio))) if n >= 3 else max(1, n - 1)
        n_val = int(round(n * val_ratio)) if n >= 5 else 0
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        n_test = n - n_train - n_val
        if n_test <= 0:
            if n_val > 0:
                n_val -= 1
            else:
                n_train = max(1, n_train - 1)
            n_test = n - n_train - n_val
        train_idx.extend(arr[:n_train].tolist())
        val_idx.extend(arr[n_train : n_train + n_val].tolist())
        test_idx.extend(arr[n_train + n_val :].tolist())

    return SplitIndices(
        train=np.asarray(sorted(train_idx), dtype=np.int64),
        val=np.asarray(sorted(val_idx), dtype=np.int64),
        test=np.asarray(sorted(test_idx), dtype=np.int64),
    )


def resolve_splits(index_rows: list[dict[str, str]], labels: list[str], args: argparse.Namespace) -> SplitIndices:
    split_values = [row.get("split", "all") for row in index_rows]
    has_existing = args.train_split in split_values and args.test_split in split_values
    if has_existing:
        train = np.asarray([i for i, value in enumerate(split_values) if value == args.train_split], dtype=np.int64)
        val = np.asarray([i for i, value in enumerate(split_values) if value == args.val_split], dtype=np.int64)
        test = np.asarray([i for i, value in enumerate(split_values) if value == args.test_split], dtype=np.int64)
        if train.size > 0 and test.size > 0:
            return SplitIndices(train=train, val=val, test=test)
    return stratified_random_split(labels, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)


def fit_standardizer(train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0, keepdims=True).astype(np.float32)
    std = train_features.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def apply_standardizer(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean) / std).astype(np.float32)


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth), int(pred)] += 1
    return matrix


def macro_f1_from_confusion(confusion: np.ndarray) -> float:
    f1_scores: list[float] = []
    for idx in range(confusion.shape[0]):
        tp = float(confusion[idx, idx])
        fp = float(confusion[:, idx].sum() - tp)
        fn = float(confusion[idx, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        if precision + recall == 0.0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def accuracy_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean()) if y_true.size else 0.0


def build_class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor | None:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    counts = np.where(counts <= 0, 1.0, counts)
    weights = counts.sum() / (num_classes * counts)
    return torch.from_numpy(weights.astype(np.float32))


def evaluate_classifier(model: nn.Module, features: np.ndarray, targets: np.ndarray, device: torch.device) -> dict[str, object]:
    if features.size == 0:
        return {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0, "confusion": np.zeros((0, 0), dtype=np.int64), "pred": np.array([], dtype=np.int64)}
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(features).to(device))
        pred = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
    confusion = confusion_matrix_np(targets, pred, logits.shape[1])
    return {
        "accuracy": accuracy_np(targets, pred),
        "macro_f1": macro_f1_from_confusion(confusion),
        "confusion": confusion,
        "pred": pred,
    }


def train_classification_probe(features: np.ndarray, index_rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    labels = [row.get(args.label_key, row.get("label", "")) for row in index_rows]
    splits = resolve_splits(index_rows, labels, args)
    class_names = sorted({labels[idx] for idx in splits.train.tolist()})
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = np.asarray([class_to_idx[label] for label in labels], dtype=np.int64)

    train_x = features[splits.train]
    val_x = features[splits.val] if splits.val.size else np.zeros((0, features.shape[1]), dtype=np.float32)
    test_x = features[splits.test]
    if args.standardize:
        mean, std = fit_standardizer(train_x)
        features = apply_standardizer(features, mean, std)
        train_x = features[splits.train]
        val_x = features[splits.val] if splits.val.size else np.zeros((0, features.shape[1]), dtype=np.float32)
        test_x = features[splits.test]
    else:
        mean = np.zeros((1, features.shape[1]), dtype=np.float32)
        std = np.ones((1, features.shape[1]), dtype=np.float32)

    train_y = y[splits.train]
    val_y = y[splits.val] if splits.val.size else np.zeros((0,), dtype=np.int64)
    test_y = y[splits.test]

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = nn.Linear(features.shape[1], len(class_names)).to(device)
    class_weight = None
    if args.class_weight == "balanced":
        class_weight = build_class_weights(train_y, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = DataLoader(train_dataset, batch_size=min(args.batch_size, len(train_dataset)), shuffle=True)

    best_state = None
    best_score = -1.0
    best_epoch = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * batch_x.size(0)
        train_metrics = evaluate_classifier(model, train_x, train_y, device)
        val_metrics = evaluate_classifier(model, val_x, val_y, device) if splits.val.size else train_metrics
        score = float(val_metrics["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(1, len(train_dataset)),
                "train_acc": float(train_metrics["accuracy"]),
                "val_acc": float(val_metrics["accuracy"]),
                "val_macro_f1": score,
            }
        )
        if score >= best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)

    train_metrics = evaluate_classifier(model, train_x, train_y, device)
    val_metrics = evaluate_classifier(model, val_x, val_y, device) if splits.val.size else train_metrics
    test_metrics = evaluate_classifier(model, test_x, test_y, device)

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "confusion_matrix.npy", test_metrics["confusion"])
    with (args.output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "task": "classification",
                "feature_key": args.feature_key,
                "best_epoch": best_epoch,
                "train": {"accuracy": train_metrics["accuracy"], "macro_f1": train_metrics["macro_f1"], "num_samples": int(splits.train.size)},
                "val": {"accuracy": val_metrics["accuracy"], "macro_f1": val_metrics["macro_f1"], "num_samples": int(splits.val.size)},
                "test": {"accuracy": test_metrics["accuracy"], "macro_f1": test_metrics["macro_f1"], "num_samples": int(splits.test.size)},
            },
            handle,
            indent=2,
        )
    with (args.output / "train_log.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
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
    torch.save(
        {
            "state_dict": best_state,
            "feature_key": args.feature_key,
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "mean": mean,
            "std": std,
            "best_epoch": best_epoch,
        },
        args.output / "probe_model.pt",
    )
    print(f"saved classification probe outputs to {args.output}")


def read_pair_rows(pair_file: Path) -> list[dict[str, str]]:
    if pair_file.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in pair_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return read_csv_rows(pair_file)


def parse_pair_label(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return 1 if int(value) != 0 else 0
    if isinstance(value, float):
        if np.isnan(value):
            return None
        return 1 if int(value) != 0 else 0

    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "t", "yes", "y", "same", "match", "positive"}:
        return 1
    if text in {"0", "false", "f", "no", "n", "different", "mismatch", "negative"}:
        return 0
    try:
        return 1 if int(text) != 0 else 0
    except ValueError:
        return None


def resolve_pair_ids(row: dict[str, str], args: argparse.Namespace) -> tuple[str, str] | None:
    for key_a, key_b in [
        (args.id_a_key, args.id_b_key),
        ("sample_id_a", "sample_id_b"),
    ]:
        sample_a = str(row.get(key_a, "")).strip()
        sample_b = str(row.get(key_b, "")).strip()
        if sample_a and sample_b:
            return sample_a, sample_b
    return None


def resolve_pair_label(row: dict[str, str], args: argparse.Namespace) -> int:
    for key in [args.pair_label_key, "pair_label", "label"]:
        if not key:
            continue
        parsed = parse_pair_label(row.get(key))
        if parsed is not None:
            return parsed
    return 0


def cosine_scores(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.sum(a * b, axis=1)
    denom = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-8)
    return (dot / denom).astype(np.float32)


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    ranks = rankdata(scores)
    sum_pos = ranks[pos].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_curve_points(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thresholds = np.unique(scores)[::-1]
    fprs = [0.0]
    tprs = [0.0]
    thresh = [thresholds[0] + 1.0 if thresholds.size else 1.0]
    pos = labels == 1
    neg = labels == 0
    n_pos = max(1, int(pos.sum()))
    n_neg = max(1, int(neg.sum()))
    for threshold in thresholds:
        pred = scores >= threshold
        tp = int(np.logical_and(pred, pos).sum())
        fp = int(np.logical_and(pred, neg).sum())
        fprs.append(fp / n_neg)
        tprs.append(tp / n_pos)
        thresh.append(float(threshold))
    fprs.append(1.0)
    tprs.append(1.0)
    thresh.append(float(thresholds[-1] - 1.0) if thresholds.size else -1.0)
    return np.asarray(fprs, dtype=np.float32), np.asarray(tprs, dtype=np.float32), np.asarray(thresh, dtype=np.float32)


def tar_at_far(labels: np.ndarray, scores: np.ndarray, far_target: float) -> float:
    fprs, tprs, _ = roc_curve_points(labels, scores)
    valid = tprs[fprs <= far_target]
    return float(valid.max()) if valid.size else 0.0


def threshold_best_accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    thresholds = np.unique(scores)
    if thresholds.size == 0:
        return 0.0
    best_threshold = float(thresholds[0])
    best_accuracy = -1.0
    for threshold in thresholds:
        pred = (scores >= threshold).astype(np.int64)
        acc = accuracy_np(labels.astype(np.int64), pred)
        if acc >= best_accuracy:
            best_accuracy = acc
            best_threshold = float(threshold)
    return best_threshold


def split_pair_rows(pair_rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    split_values = [row.get("split", "all") for row in pair_rows]
    if args.train_split in split_values or args.test_split in split_values:
        train = [row for row in pair_rows if row.get("split", "all") == args.train_split]
        val = [row for row in pair_rows if row.get("split", "all") == args.val_split]
        test = [row for row in pair_rows if row.get("split", "all") == args.test_split]
        if train and test:
            return train, val, test
    rng = np.random.default_rng(args.seed)
    shuffled = pair_rows.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(round(n * args.train_ratio)))
    n_val = int(round(n * args.val_ratio))
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]


def pair_arrays(
    pair_rows: list[dict[str, str]],
    feature_lookup: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    ids: list[tuple[str, str]] = []
    for row in pair_rows:
        resolved = resolve_pair_ids(row, args)
        if resolved is None:
            continue
        sample_a, sample_b = resolved
        if sample_a not in feature_lookup or sample_b not in feature_lookup:
            continue
        features.append(np.abs(feature_lookup[sample_a] - feature_lookup[sample_b]))
        labels.append(resolve_pair_label(row, args))
        ids.append((sample_a, sample_b))
    if not features:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64), ids
    return np.stack(features).astype(np.float32), np.asarray(labels, dtype=np.int64), ids


def train_linear_pair_probe(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, args: argparse.Namespace) -> tuple[nn.Module, list[dict[str, float]], int]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = nn.Linear(train_x.shape[1], 1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32))), batch_size=min(args.batch_size, len(train_x)), shuffle=True)

    best_state = None
    best_auc = -1.0
    best_epoch = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * batch_x.size(0)

        model.eval()
        with torch.inference_mode():
            val_logits = model(torch.from_numpy(val_x).to(device)).squeeze(1).cpu().numpy() if len(val_x) else model(torch.from_numpy(train_x).to(device)).squeeze(1).cpu().numpy()
        target = val_y if len(val_y) else train_y
        score = auc_score(target, val_logits)
        history.append({"epoch": epoch, "train_loss": running_loss / max(1, len(train_x)), "val_auc": score})
        if score >= best_auc:
            best_auc = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Linear pair probe training failed.")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def evaluate_verification(scores: np.ndarray, labels: np.ndarray, threshold: float, far_target: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(np.int64)
    return {
        "accuracy": accuracy_np(labels, pred),
        "auc": auc_score(labels, scores),
        "tar_at_far": tar_at_far(labels, scores, far_target),
        "threshold": float(threshold),
    }


def run_verification(features: np.ndarray, index_rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    if args.pair_file is None:
        raise ValueError("--pair-file is required for verification tasks.")
    feature_lookup = {row["sample_id"]: features[idx] for idx, row in enumerate(index_rows)}
    pair_rows = read_pair_rows(args.pair_file)
    train_rows, val_rows, test_rows = split_pair_rows(pair_rows, args)

    if args.metric == "cosine":
        def scores_for(rows_subset: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
            left = []
            right = []
            labels = []
            ids = []
            for row in rows_subset:
                resolved = resolve_pair_ids(row, args)
                if resolved is None:
                    continue
                a, b = resolved
                if a not in feature_lookup or b not in feature_lookup:
                    continue
                left.append(feature_lookup[a])
                right.append(feature_lookup[b])
                labels.append(resolve_pair_label(row, args))
                ids.append((a, b))
            if not left:
                return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64), ids
            return cosine_scores(np.stack(left).astype(np.float32), np.stack(right).astype(np.float32)), np.asarray(labels, dtype=np.int64), ids

        train_scores, train_labels, _ = scores_for(train_rows)
        val_scores, val_labels, _ = scores_for(val_rows)
        test_scores, test_labels, test_ids = scores_for(test_rows)
        threshold_source_scores = val_scores if val_scores.size else train_scores
        threshold_source_labels = val_labels if val_labels.size else train_labels
        threshold = threshold_best_accuracy(threshold_source_labels, threshold_source_scores)
        metrics = {
            "task": "verification",
            "feature_key": args.feature_key,
            "train": evaluate_verification(train_scores, train_labels, threshold, args.far_target),
            "val": evaluate_verification(val_scores, val_labels, threshold, args.far_target) if val_scores.size else {"accuracy": 0.0, "auc": 0.0, "tar_at_far": 0.0, "threshold": float(threshold)},
            "test": evaluate_verification(test_scores, test_labels, threshold, args.far_target),
            "metric": "cosine",
        }
        logits_for_test = test_scores
        best_epoch = 0
        history: list[dict[str, float]] = []
    else:
        train_x, train_y, _ = pair_arrays(train_rows, feature_lookup, args)
        val_x, val_y, _ = pair_arrays(val_rows, feature_lookup, args)
        test_x, test_y, test_ids = pair_arrays(test_rows, feature_lookup, args)
        if args.standardize and len(train_x):
            mean, std = fit_standardizer(train_x)
            train_x = apply_standardizer(train_x, mean, std)
            val_x = apply_standardizer(val_x, mean, std) if len(val_x) else val_x
            test_x = apply_standardizer(test_x, mean, std) if len(test_x) else test_x
        else:
            mean = np.zeros((1, features.shape[1]), dtype=np.float32)
            std = np.ones((1, features.shape[1]), dtype=np.float32)
        model, history, best_epoch = train_linear_pair_probe(train_x, train_y, val_x, val_y, args)
        device = torch.device(args.device)
        model.eval()
        with torch.inference_mode():
            train_scores = model(torch.from_numpy(train_x).to(device)).squeeze(1).cpu().numpy() if len(train_x) else np.zeros((0,), dtype=np.float32)
            val_scores = model(torch.from_numpy(val_x).to(device)).squeeze(1).cpu().numpy() if len(val_x) else np.zeros((0,), dtype=np.float32)
            test_scores = model(torch.from_numpy(test_x).to(device)).squeeze(1).cpu().numpy() if len(test_x) else np.zeros((0,), dtype=np.float32)
        threshold_source_scores = val_scores if val_scores.size else train_scores
        threshold_source_labels = val_y if val_scores.size else train_y
        threshold = threshold_best_accuracy(threshold_source_labels, threshold_source_scores)
        metrics = {
            "task": "verification",
            "feature_key": args.feature_key,
            "train": evaluate_verification(train_scores, train_y, threshold, args.far_target),
            "val": evaluate_verification(val_scores, val_y, threshold, args.far_target) if val_scores.size else {"accuracy": 0.0, "auc": 0.0, "tar_at_far": 0.0, "threshold": float(threshold)},
            "test": evaluate_verification(test_scores, test_y, threshold, args.far_target),
            "metric": "linear_probe",
            "best_epoch": best_epoch,
        }
        logits_for_test = test_scores
        torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std}, args.output / "probe_model.pt")
        test_labels = test_y

    args.output.mkdir(parents=True, exist_ok=True)
    if args.metric == "cosine":
        test_scores, test_labels, test_ids = scores_for(test_rows)
        fprs, tprs, thresholds = roc_curve_points(test_labels, test_scores)
    else:
        fprs, tprs, thresholds = roc_curve_points(test_labels, logits_for_test)

    with (args.output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (args.output / "train_log.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with (args.output / "roc_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["threshold", "fpr", "tpr"])
        writer.writeheader()
        for threshold, fpr, tpr in zip(thresholds.tolist(), fprs.tolist(), tprs.tolist()):
            writer.writerow({"threshold": threshold, "fpr": fpr, "tpr": tpr})
    with (args.output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id_a", "sample_id_b", "pair_label", "score"])
        writer.writeheader()
        for (sample_a, sample_b), label, score in zip(test_ids, test_labels.tolist(), logits_for_test.tolist()):
            writer.writerow({"sample_id_a": sample_a, "sample_id_b": sample_b, "pair_label": label, "score": score})
    print(f"saved verification outputs to {args.output}")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    features, index_rows = load_features(args.feature_root, args.feature_key)
    if args.standardize and args.task == "verification" and args.metric == "cosine":
        pass
    if args.task == "classification":
        train_classification_probe(features, index_rows, args)
    else:
        run_verification(features, index_rows, args)


if __name__ == "__main__":
    main()


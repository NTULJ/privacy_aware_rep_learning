from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
cache_dir = PROJECT_ROOT / ".matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


SKIPPED_EXIT_CODE = 3


class SkipPlot(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot probe training curves from train_log.json and metrics.json.")
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"failed to parse JSON: {path}") from exc


def normalize_history(history: object, metrics: dict[str, object]) -> tuple[list[int], list[float], list[float], str, str]:
    if not isinstance(history, list):
        raise ValueError("train_log.json must contain a JSON array.")
    if not history:
        if metrics.get("metric") == "cosine":
            raise SkipPlot("metric=cosine has no epoch training log")
        if metrics.get("reused_model") is True:
            raise SkipPlot("reused utility head has no training log")
        raise SkipPlot("train_log.json is empty")

    if not all(isinstance(row, dict) for row in history):
        raise ValueError("train_log.json entries must be JSON objects.")

    first_row = history[0]
    if "val_auc" in first_row:
        metric_key = "val_auc"
        metric_label = "val_auc"
        metric_name = str(metrics.get("metric", "verification"))
        task_label = f"verification ({metric_name})"
    elif "val_macro_f1" in first_row:
        metric_key = "val_macro_f1"
        metric_label = "val_macro_f1"
        task_label = "classification"
    else:
        raise ValueError("train_log.json entries must include either val_auc or val_macro_f1.")

    epochs: list[int] = []
    losses: list[float] = []
    metric_values: list[float] = []
    for idx, row in enumerate(history, start=1):
        missing = [key for key in ("epoch", "train_loss", metric_key) if key not in row]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"train_log.json entry {idx} is missing required keys: {joined}")
        try:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["train_loss"]))
            metric_values.append(float(row[metric_key]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"train_log.json entry {idx} contains non-numeric curve values.") from exc

    return epochs, losses, metric_values, metric_label, task_label


def metrics_title(metrics: dict[str, object], task_label: str) -> str:
    feature_key = str(metrics.get("feature_key", "unknown_feature"))
    return f"{feature_key} | {task_label}"


def plot_curves(
    output_path: Path,
    epochs: list[int],
    losses: list[float],
    metric_values: list[float],
    metric_label: str,
    title: str,
    best_epoch: int | None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to render training curves") from exc

    fig, metric_ax = plt.subplots(figsize=(9, 5.4))
    loss_ax = metric_ax.twinx()

    metric_line = metric_ax.plot(
        epochs,
        metric_values,
        color="#0b7285",
        marker="o",
        linewidth=2.0,
        label=metric_label,
    )
    loss_line = loss_ax.plot(
        epochs,
        losses,
        color="#c92a2a",
        marker="s",
        linewidth=1.8,
        alpha=0.85,
        label="train_loss",
    )

    if best_epoch is not None:
        metric_ax.axvline(best_epoch, color="#495057", linestyle="--", linewidth=1.2, alpha=0.8)
        metric_ax.text(
            best_epoch,
            0.02,
            f"best_epoch={best_epoch}",
            transform=metric_ax.get_xaxis_transform(),
            rotation=90,
            va="bottom",
            ha="right",
            color="#495057",
        )

    metric_ax.set_xlabel("epoch")
    metric_ax.set_ylabel(metric_label)
    loss_ax.set_ylabel("train_loss")
    metric_ax.set_title(title)
    metric_ax.grid(True, alpha=0.3)

    handles = metric_line + loss_line
    labels = [line.get_label() for line in handles]
    metric_ax.legend(handles, labels, loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    probe_root = args.probe_root.resolve()
    output_path = args.output.resolve() if args.output is not None else probe_root / "training_curve.png"

    metrics_path = probe_root / "metrics.json"
    train_log_path = probe_root / "train_log.json"

    metrics = read_json(metrics_path)
    if not isinstance(metrics, dict):
        raise ValueError("metrics.json must contain a JSON object.")

    history = read_json(train_log_path)
    epochs, losses, metric_values, metric_label, task_label = normalize_history(history, metrics)

    best_epoch_raw = metrics.get("best_epoch")
    best_epoch = int(best_epoch_raw) if isinstance(best_epoch_raw, (int, float)) else None
    title = metrics_title(metrics, task_label)
    plot_curves(output_path, epochs, losses, metric_values, metric_label, title, best_epoch)
    print(str(output_path))


if __name__ == "__main__":
    try:
        main()
    except SkipPlot as exc:
        print(str(exc))
        raise SystemExit(SKIPPED_EXIT_CODE)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)


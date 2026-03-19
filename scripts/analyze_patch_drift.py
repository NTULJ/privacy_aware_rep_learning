from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for env_name, relative_path in {
    "MPLCONFIGDIR": ".matplotlib",
}.items():
    cache_dir = PROJECT_ROOT / relative_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(env_name, str(cache_dir))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze patch-level drift between clean and privacy-protected Qwen3-VL token features.")
    parser.add_argument("--feature-root", type=Path, required=True, help="Feature extraction directory created by extract_qwen_vl_features.py")
    parser.add_argument("--stage-clean", default="x_pre_clean")
    parser.add_argument("--stage-priv", default="x_pre_priv")
    parser.add_argument("--split", default="all")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corr-method", choices=("spearman", "pearson"), default="spearman")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--num-visualizations", type=int, default=6)
    return parser.parse_args()


def read_index(index_path: Path) -> list[dict[str, str]]:
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    if method == "pearson":
        value = pearsonr(x, y)[0]
    else:
        value = spearmanr(x, y)[0]
    if value is None or math.isnan(value):
        return 0.0
    return float(value)


def cosine_drift(clean: np.ndarray, priv: np.ndarray) -> np.ndarray:
    dot = np.sum(clean * priv, axis=1)
    clean_norm = np.linalg.norm(clean, axis=1)
    priv_norm = np.linalg.norm(priv, axis=1)
    denom = np.maximum(clean_norm * priv_norm, 1e-8)
    cosine = np.clip(dot / denom, -1.0, 1.0)
    return 1.0 - cosine


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if mask.any():
        return float(values[mask].mean())
    return float(values.mean())


def masked_ratio(values: np.ndarray, mask: np.ndarray) -> float:
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    return float(values[mask].sum() / total) if mask.any() else 0.0


def flatten_for_stage(patch_data, stage_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if stage_name.startswith("hpool"):
        scores = patch_data["patch_scores_merged"].astype(np.float32).reshape(-1)
        head = patch_data["head_patch_mask_merged"].astype(bool).reshape(-1)
        torso = patch_data["torso_patch_mask_merged"].astype(bool).reshape(-1)
        human = patch_data["human_patch_mask_merged"].astype(bool).reshape(-1)
        background = patch_data["background_patch_mask_merged"].astype(bool).reshape(-1)
    else:
        scores = patch_data["patch_scores"].astype(np.float32).reshape(-1)
        head = patch_data["head_patch_mask"].astype(bool).reshape(-1)
        torso = patch_data["torso_patch_mask"].astype(bool).reshape(-1)
        human = patch_data["human_patch_mask"].astype(bool).reshape(-1)
        background = patch_data["background_patch_mask"].astype(bool).reshape(-1)
    return scores, head, torso, human, background


def drift_overlay(image_path: Path, drift_grid: np.ndarray, destination: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    heatmap_grid = cv2.resize(drift_grid.astype(np.float32), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    if heatmap_grid.max() > 0:
        heatmap_grid = heatmap_grid / heatmap_grid.max()
    heatmap = cv2.applyColorMap(np.clip(heatmap_grid * 255.0, 0.0, 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(image, 0.60, heatmap, 0.40, 0.0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), overlay)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summary_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_index(args.feature_root / "index.csv")
    if args.split != "all":
        rows = [row for row in rows if row.get("split", "all") == args.split]
    if args.sample_limit is not None:
        rows = rows[: args.sample_limit]

    metrics_rows: list[dict[str, object]] = []
    visualization_rows: list[tuple[float, dict[str, object], np.ndarray]] = []

    for row in rows:
        token_path = row.get("token_path", "")
        patch_path = row.get("patch_path", "")
        if not token_path or not patch_path:
            continue
        token_file = args.feature_root / token_path
        patch_file = args.feature_root / patch_path
        if not token_file.exists() or not patch_file.exists():
            continue

        with np.load(token_file) as token_data, np.load(patch_file) as patch_data:
            if args.stage_clean not in token_data or args.stage_priv not in token_data:
                continue
            clean = token_data[args.stage_clean].astype(np.float32)
            priv = token_data[args.stage_priv].astype(np.float32)
            if clean.shape != priv.shape:
                continue

            patch_scores, head_mask, torso_mask, human_mask, background_mask = flatten_for_stage(patch_data, args.stage_clean)
            if clean.shape[0] != patch_scores.size:
                continue

            l2 = np.linalg.norm(priv - clean, axis=1)
            cos = cosine_drift(clean, priv)
            head_energy_ratio = masked_ratio(l2, head_mask)
            torso_energy_ratio = masked_ratio(l2, torso_mask)
            bg_energy_ratio = masked_ratio(l2, background_mask)

            metric_row = {
                "sample_id": row["sample_id"],
                "image_path": row["image_path"],
                "num_patches": int(clean.shape[0]),
                "corr_l2": safe_correlation(patch_scores, l2, args.corr_method),
                "corr_cos": safe_correlation(patch_scores, cos, args.corr_method),
                "head_mean_l2": masked_mean(l2, head_mask),
                "torso_mean_l2": masked_mean(l2, torso_mask),
                "bg_mean_l2": masked_mean(l2, background_mask),
                "head_mean_cos": masked_mean(cos, head_mask),
                "torso_mean_cos": masked_mean(cos, torso_mask),
                "bg_mean_cos": masked_mean(cos, background_mask),
                "head_energy_ratio": head_energy_ratio,
                "torso_energy_ratio": torso_energy_ratio,
                "bg_energy_ratio": bg_energy_ratio,
            }
            metrics_rows.append(metric_row)

            grid_key = "patch_scores_merged" if args.stage_clean.startswith("hpool") else "patch_scores"
            drift_grid = l2.reshape(patch_data[grid_key].shape)
            visualization_rows.append((head_energy_ratio, metric_row, drift_grid))

    if not metrics_rows:
        raise RuntimeError("No valid samples were found for drift analysis. Did you run extract_qwen_vl_features.py with --save-token-features?")

    fieldnames = list(metrics_rows[0].keys())
    write_csv(args.output / "per_sample_metrics.csv", metrics_rows, fieldnames=fieldnames)

    summary = {
        "stage_clean": args.stage_clean,
        "stage_priv": args.stage_priv,
        "num_samples": len(metrics_rows),
        "corr_l2": summary_stats([float(row["corr_l2"]) for row in metrics_rows]),
        "corr_cos": summary_stats([float(row["corr_cos"]) for row in metrics_rows]),
        "head_mean_l2": summary_stats([float(row["head_mean_l2"]) for row in metrics_rows]),
        "torso_mean_l2": summary_stats([float(row["torso_mean_l2"]) for row in metrics_rows]),
        "bg_mean_l2": summary_stats([float(row["bg_mean_l2"]) for row in metrics_rows]),
        "head_energy_ratio": summary_stats([float(row["head_energy_ratio"]) for row in metrics_rows]),
        "torso_energy_ratio": summary_stats([float(row["torso_energy_ratio"]) for row in metrics_rows]),
        "bg_energy_ratio": summary_stats([float(row["bg_energy_ratio"]) for row in metrics_rows]),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        f"stage_clean: {args.stage_clean}",
        f"stage_priv: {args.stage_priv}",
        f"samples: {len(metrics_rows)}",
        f"corr_l2 mean: {summary['corr_l2']['mean']:.4f}",
        f"corr_cos mean: {summary['corr_cos']['mean']:.4f}",
        f"head_mean_l2 mean: {summary['head_mean_l2']['mean']:.4f}",
        f"torso_mean_l2 mean: {summary['torso_mean_l2']['mean']:.4f}",
        f"bg_mean_l2 mean: {summary['bg_mean_l2']['mean']:.4f}",
        f"head_energy_ratio mean: {summary['head_energy_ratio']['mean']:.4f}",
    ]
    (args.output / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if args.plot:
        plots_dir = args.output / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        corr_l2_values = np.asarray([float(row["corr_l2"]) for row in metrics_rows], dtype=np.float32)
        corr_cos_values = np.asarray([float(row["corr_cos"]) for row in metrics_rows], dtype=np.float32)
        plt.figure(figsize=(8, 4))
        plt.hist(corr_l2_values, bins=20, alpha=0.8)
        plt.title("Patch-score vs L2 drift correlation")
        plt.xlabel("Correlation")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(plots_dir / "correlation_histogram_l2.png", dpi=180)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.hist(corr_cos_values, bins=20, alpha=0.8)
        plt.title("Patch-score vs cosine drift correlation")
        plt.xlabel("Correlation")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(plots_dir / "correlation_histogram_cos.png", dpi=180)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.hist([float(row["head_mean_l2"]) for row in metrics_rows], bins=20, alpha=0.8)
        plt.title("Head-patch L2 drift")
        plt.xlabel("Mean L2 drift")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(plots_dir / "drift_histogram.png", dpi=180)
        plt.close()

        group_means = [
            summary["head_mean_l2"]["mean"],
            summary["torso_mean_l2"]["mean"],
            summary["bg_mean_l2"]["mean"],
        ]
        plt.figure(figsize=(6, 4))
        plt.bar(["head", "torso", "background"], group_means)
        plt.title("Mean L2 drift by region")
        plt.ylabel("Mean L2 drift")
        plt.tight_layout()
        plt.savefig(plots_dir / "group_bar.png", dpi=180)
        plt.close()

        for _, metric_row, drift_grid in sorted(visualization_rows, key=lambda item: item[0], reverse=True)[: args.num_visualizations]:
            image_path = Path(str(metric_row["image_path"]))
            destination = plots_dir / f"sample_{metric_row['sample_id']}_drift_overlay.png"
            drift_overlay(image_path, drift_grid, destination)

    print(f"saved drift analysis to {args.output}")


if __name__ == "__main__":
    main()

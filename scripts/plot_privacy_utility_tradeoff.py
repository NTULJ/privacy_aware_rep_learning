from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cross-dataset privacy-utility tradeoff curves from Stanford40 and LFW summaries."
    )
    parser.add_argument("--stanford-ours-summary", type=Path, default=None)
    parser.add_argument("--lfw-ours-summary", type=Path, default=None)
    parser.add_argument("--stanford-uniform-summary", type=Path, default=None)
    parser.add_argument("--lfw-uniform-summary", type=Path, default=None)
    parser.add_argument("--stanford-stage-summary", type=Path, default=None)
    parser.add_argument("--lfw-stage-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stanford-method-metric",
        default="results.hpool_priv_global.test.accuracy",
        help="Nested metric path for method comparison x-axis.",
    )
    parser.add_argument(
        "--lfw-method-metric",
        default="results.priv_verification.test.auc",
        help="Nested metric path for method comparison y-axis.",
    )
    parser.add_argument(
        "--stanford-stage-metric",
        default="hpool_priv_global_test_acc",
        help="Flat key from stage_sweep_summary.json for x-axis.",
    )
    parser.add_argument(
        "--lfw-stage-metric",
        default="priv_verification_test_auc",
        help="Flat key from stage_sweep_summary.json for y-axis.",
    )
    parser.add_argument("--title-prefix", default="Privacy-Utility Tradeoff")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def get_nested(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_points(
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    label_key: str,
    out_path: Path,
    title: str,
    connect: bool = False,
) -> None:
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    xs = [float(row[x_key]) for row in rows]
    ys = [float(row[y_key]) for row in rows]
    labels = [str(row[label_key]) for row in rows]

    ax.scatter(xs, ys, s=70)
    if connect and len(rows) > 1:
        ax.plot(xs, ys, linewidth=1.5, alpha=0.7)

    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6))

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_method_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = [
        ("ours", args.stanford_ours_summary, args.lfw_ours_summary),
        ("uniform", args.stanford_uniform_summary, args.lfw_uniform_summary),
    ]
    for label, stanford_path, lfw_path in specs:
        if stanford_path is None or lfw_path is None:
            continue
        if not stanford_path.exists() or not lfw_path.exists():
            continue
        stanford_payload = read_json(stanford_path)
        lfw_payload = read_json(lfw_path)
        rows.append(
            {
                "label": label,
                "stanford_metric": get_nested(stanford_payload, args.stanford_method_metric),
                "lfw_metric": get_nested(lfw_payload, args.lfw_method_metric),
                "stanford_summary": str(stanford_path),
                "lfw_summary": str(lfw_path),
            }
        )
    return [row for row in rows if row["stanford_metric"] is not None and row["lfw_metric"] is not None]


def build_stage_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.stanford_stage_summary is None or args.lfw_stage_summary is None:
        return []
    if not args.stanford_stage_summary.exists() or not args.lfw_stage_summary.exists():
        return []

    stanford_rows = {
        str(row.get("injection_stage")): row
        for row in read_json(args.stanford_stage_summary)
        if row.get("injection_stage") is not None
    }
    lfw_rows = {
        str(row.get("injection_stage")): row
        for row in read_json(args.lfw_stage_summary)
        if row.get("injection_stage") is not None
    }

    ordered_stages = ["x_pre", "block1", "block8", "block16", "block24", "hpool"]
    rows: list[dict[str, Any]] = []
    for stage in ordered_stages:
        if stage not in stanford_rows or stage not in lfw_rows:
            continue
        s_row = stanford_rows[stage]
        l_row = lfw_rows[stage]
        rows.append(
            {
                "injection_stage": stage,
                "stanford_metric": s_row.get(args.stanford_stage_metric),
                "lfw_metric": l_row.get(args.lfw_stage_metric),
                "stanford_summary": s_row.get("summary_path", ""),
                "lfw_summary": l_row.get("summary_path", ""),
            }
        )
    return [row for row in rows if row["stanford_metric"] is not None and row["lfw_metric"] is not None]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    method_rows = build_method_rows(args)
    if method_rows:
        method_csv = args.output_dir / "method_tradeoff.csv"
        write_csv(
            method_csv,
            method_rows,
            headers=["label", "stanford_metric", "lfw_metric", "stanford_summary", "lfw_summary"],
        )
        plot_points(
            method_rows,
            x_key="stanford_metric",
            y_key="lfw_metric",
            label_key="label",
            out_path=args.output_dir / "method_tradeoff.png",
            title=f"{args.title_prefix}: Ours vs Uniform",
            connect=False,
        )

    stage_rows = build_stage_rows(args)
    if stage_rows:
        stage_csv = args.output_dir / "stage_tradeoff.csv"
        write_csv(
            stage_csv,
            stage_rows,
            headers=["injection_stage", "stanford_metric", "lfw_metric", "stanford_summary", "lfw_summary"],
        )
        plot_points(
            stage_rows,
            x_key="stanford_metric",
            y_key="lfw_metric",
            label_key="injection_stage",
            out_path=args.output_dir / "stage_tradeoff.png",
            title=f"{args.title_prefix}: Injection Stage Sweep",
            connect=True,
        )

    summary = {
        "output_dir": str(args.output_dir),
        "method_tradeoff_csv": str(args.output_dir / "method_tradeoff.csv") if method_rows else "",
        "method_tradeoff_png": str(args.output_dir / "method_tradeoff.png") if method_rows else "",
        "stage_tradeoff_csv": str(args.output_dir / "stage_tradeoff.csv") if stage_rows else "",
        "stage_tradeoff_png": str(args.output_dir / "stage_tradeoff.png") if stage_rows else "",
        "num_method_points": len(method_rows),
        "num_stage_points": len(stage_rows),
        "stanford_method_metric": args.stanford_method_metric,
        "lfw_method_metric": args.lfw_method_metric,
        "stanford_stage_metric": args.stanford_stage_metric,
        "lfw_stage_metric": args.lfw_stage_metric,
    }
    summary_path = args.output_dir / "tradeoff_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

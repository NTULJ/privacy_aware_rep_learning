from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for env_name, relative_path in {
    "YOLO_CONFIG_DIR": ".ultralytics",
    "MPLCONFIGDIR": ".matplotlib",
    "TORCH_HOME": ".torch",
}.items():
    cache_dir = PROJECT_ROOT / relative_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(env_name, str(cache_dir))

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect people with a pretrained YOLO model.")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to an image, a directory of images, or a video file.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Ultralytics model name or local weight path.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device, e.g. cpu, 0, 0,1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/person_detection"),
        help="Directory to save rendered outputs.",
    )
    return parser.parse_args()


def iter_sources(source: Path) -> list[Path]:
    if source.is_dir():
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        files: list[Path] = []
        for pattern in patterns:
            files.extend(sorted(source.glob(pattern)))
        return files
    return [source]


def save_rendered(result, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = result.plot()
    cv2.imwrite(str(destination), rendered)


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    sources = iter_sources(args.source)
    if not sources:
        raise FileNotFoundError(f"No input files found in {args.source}")

    for src in sources:
        results = model.predict(
            source=str(src),
            classes=[0],
            conf=args.conf,
            device=args.device,
            verbose=False,
        )
        if not results:
            continue

        destination = args.output / src.name
        save_rendered(results[0], destination)

        boxes = results[0].boxes
        person_count = 0 if boxes is None else len(boxes)
        print(f"{src}: detected {person_count} person(s), saved to {destination}")


if __name__ == "__main__":
    main()

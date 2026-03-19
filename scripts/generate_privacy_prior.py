from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
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
import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0
    source: str = ""

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center_x(self) -> float:
        return self.x1 + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y1 + self.height / 2.0

    def clip(self, width: int, height: int) -> "BoundingBox | None":
        x1 = min(max(self.x1, 0.0), float(width))
        y1 = min(max(self.y1, 0.0), float(height))
        x2 = min(max(self.x2, 0.0), float(width))
        y2 = min(max(self.y2, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            return None
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, conf=self.conf, source=self.source)

    def scale(self, scale_x: float, scale_y: float) -> "BoundingBox":
        return BoundingBox(
            x1=self.x1 * scale_x,
            y1=self.y1 * scale_y,
            x2=self.x2 * scale_x,
            y2=self.y2 * scale_y,
            conf=self.conf,
            source=self.source,
        )

    def to_dict(self) -> dict[str, float | str]:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "conf": round(self.conf, 4),
            "source": self.source,
        }


@dataclass(frozen=True)
class DetectionRecord:
    person_box: BoundingBox
    head_box: BoundingBox
    head_core_box: BoundingBox
    torso_box: BoundingBox
    head_source: str
    aux_box: BoundingBox | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "person_box": self.person_box.to_dict(),
            "head_box": self.head_box.to_dict(),
            "head_core_box": self.head_core_box.to_dict(),
            "torso_box": self.torso_box.to_dict(),
            "head_source": self.head_source,
            "aux_box": None if self.aux_box is None else self.aux_box.to_dict(),
        }


class OpenCvFaceDetector:
    def __init__(self) -> None:
        frontal_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        profile_path = Path(cv2.data.haarcascades) / "haarcascade_profileface.xml"

        self.frontal = cv2.CascadeClassifier(str(frontal_path))
        self.profile = cv2.CascadeClassifier(str(profile_path)) if profile_path.exists() else None

    @property
    def available(self) -> bool:
        return not self.frontal.empty()

    def detect(self, image: np.ndarray, person_box: BoundingBox) -> BoundingBox | None:
        roi = crop_image(image, person_box)
        if roi is None:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        roi_height, roi_width = gray.shape[:2]
        min_dim = min(roi_width, roi_height)
        if min_dim < 48:
            scale = 3.0
        elif min_dim < 96:
            scale = 2.0
        else:
            scale = 1.0

        work_gray = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        work_height, work_width = work_gray.shape[:2]
        min_size = max(18, int(min(work_width, work_height) * 0.10))

        candidates: list[BoundingBox] = []
        frontal_boxes = self.frontal.detectMultiScale(
            work_gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(min_size, min_size),
        )
        candidates.extend(local_boxes_to_global_scaled(frontal_boxes, person_box, scale=scale, source="cv2_frontalface"))

        if self.profile is not None and not self.profile.empty():
            profile_boxes = self.profile.detectMultiScale(
                work_gray,
                scaleFactor=1.08,
                minNeighbors=3,
                minSize=(min_size, min_size),
            )
            candidates.extend(local_boxes_to_global_scaled(profile_boxes, person_box, scale=scale, source="cv2_profileface"))

            flipped = cv2.flip(work_gray, 1)
            mirrored_boxes = self.profile.detectMultiScale(
                flipped,
                scaleFactor=1.08,
                minNeighbors=3,
                minSize=(min_size, min_size),
            )
            for x, y, w, h in mirrored_boxes:
                mirrored_x = work_width - x - w
                candidates.append(
                    BoundingBox(
                        x1=person_box.x1 + mirrored_x / scale,
                        y1=person_box.y1 + y / scale,
                        x2=person_box.x1 + (mirrored_x + w) / scale,
                        y2=person_box.y1 + (y + h) / scale,
                        conf=1.0,
                        source="cv2_profileface_mirrored",
                    )
                )

        return select_best_auxiliary_box(candidates, person_box)


class YuNetFaceDetector:
    def __init__(
        self,
        model_path: Path,
        score_threshold: float = 0.55,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        self.model_path = Path(model_path)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.detector = None
        if hasattr(cv2, "FaceDetectorYN_create") and self.model_path.exists():
            self.detector = cv2.FaceDetectorYN_create(
                str(self.model_path),
                "",
                (320, 320),
                self.score_threshold,
                self.nms_threshold,
                self.top_k,
            )

    @property
    def available(self) -> bool:
        return self.detector is not None

    def _detect_faces(self, work_image: np.ndarray) -> np.ndarray | None:
        if self.detector is None:
            return None
        self.detector.setInputSize((int(work_image.shape[1]), int(work_image.shape[0])))
        _, faces = self.detector.detect(work_image)
        return faces

    def detect(self, image: np.ndarray, person_box: BoundingBox) -> BoundingBox | None:
        roi = crop_image(image, person_box)
        if roi is None or self.detector is None:
            return None

        roi_height, roi_width = roi.shape[:2]
        min_dim = min(roi_width, roi_height)
        if min_dim <= 0:
            return None

        scale = min(4.0, max(1.0, 96.0 / float(min_dim)))
        if min_dim < 40:
            scale = max(scale, 3.2)
        elif min_dim < 72:
            scale = max(scale, 1.8)

        work_image = (
            cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            if scale > 1.0
            else roi
        )
        work_width = work_image.shape[1]

        candidates: list[BoundingBox] = []
        faces = self._detect_faces(work_image)
        if faces is not None:
            for row in faces:
                x, y, w, h = row[:4]
                confidence = float(row[-1])
                candidates.append(
                    BoundingBox(
                        x1=person_box.x1 + float(x) / scale,
                        y1=person_box.y1 + float(y) / scale,
                        x2=person_box.x1 + float(x + w) / scale,
                        y2=person_box.y1 + float(y + h) / scale,
                        conf=confidence,
                        source="yunet_face",
                    )
                )

        flipped = cv2.flip(work_image, 1)
        mirrored_faces = self._detect_faces(flipped)
        if mirrored_faces is not None:
            for row in mirrored_faces:
                x, y, w, h = row[:4]
                mirrored_x = float(work_width - x - w)
                confidence = float(row[-1])
                candidates.append(
                    BoundingBox(
                        x1=person_box.x1 + mirrored_x / scale,
                        y1=person_box.y1 + float(y) / scale,
                        x2=person_box.x1 + (mirrored_x + float(w)) / scale,
                        y2=person_box.y1 + float(y + h) / scale,
                        conf=confidence,
                        source="yunet_face_mirrored",
                    )
                )

        return select_best_auxiliary_box(candidates, person_box)


class YoloAuxDetector:
    def __init__(self, model_path: str, conf: float, device: str, kind: str) -> None:
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device
        self.kind = kind

    def detect(self, image: np.ndarray, person_box: BoundingBox) -> BoundingBox | None:
        roi = crop_image(image, person_box)
        if roi is None:
            return None

        results = self.model.predict(source=roi, conf=self.conf, device=self.device, verbose=False)
        if not results:
            return None

        boxes = results[0].boxes
        if boxes is None:
            return None

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        candidates: list[BoundingBox] = []
        for coords, confidence in zip(xyxy, confs):
            x1, y1, x2, y2 = coords.tolist()
            candidates.append(
                BoundingBox(
                    x1=person_box.x1 + float(x1),
                    y1=person_box.y1 + float(y1),
                    x2=person_box.x1 + float(x2),
                    y2=person_box.y1 + float(y2),
                    conf=float(confidence),
                    source=f"yolo_aux_{self.kind}",
                )
            )
        return select_best_auxiliary_box(candidates, person_box)


class FallbackFaceDetector:
    def __init__(self, detectors: list[object]) -> None:
        self.detectors = detectors

    def detect(self, image: np.ndarray, person_box: BoundingBox) -> BoundingBox | None:
        for detector in self.detectors:
            box = detector.detect(image, person_box)
            if box is not None:
                return box
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a privacy prior map and patch-level sigma values from person detections."
    )
    parser.add_argument("--source", type=Path, required=True, help="Path to an image or a directory of images.")
    parser.add_argument("--person-model", default="yolo11n.pt", help="Ultralytics YOLO model used for person detection.")
    parser.add_argument(
        "--face-model",
        default=None,
        help="Optional YOLO face/head detector. If omitted, OpenCV face cascades are used before head fallback.",
    )
    parser.add_argument(
        "--face-model-kind",
        choices=("face", "head"),
        default="face",
        help="Whether the optional auxiliary YOLO model predicts face boxes or head boxes.",
    )
    parser.add_argument("--person-conf", type=float, default=0.25, help="Confidence threshold for person detections.")
    parser.add_argument("--face-conf", type=float, default=0.20, help="Confidence threshold for the optional auxiliary detector.")
    parser.add_argument(
        "--yunet-model",
        type=Path,
        default=Path("models/face_detection_yunet_2023mar.onnx"),
        help="Optional YuNet ONNX face detector used before OpenCV cascade fallback.",
    )
    parser.add_argument(
        "--yunet-score-threshold",
        type=float,
        default=0.55,
        help="Score threshold for YuNet face detection inside each person ROI.",
    )
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or 0.")
    parser.add_argument("--output", type=Path, default=Path("runs/privacy_prior"), help="Directory where outputs are written.")
    parser.add_argument(
        "--map-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optional privacy-map size. Defaults to the input image size.",
    )
    parser.add_argument("--patch-size", type=int, default=14, help="Patch size used for patch-level aggregation.")
    parser.add_argument(
        "--patch-alpha",
        type=float,
        default=0.70,
        help="Blend factor for patch score aggregation: alpha * max + (1 - alpha) * mean.",
    )
    parser.add_argument("--sigma-min", type=float, default=0.0, help="Lower bound for patch sigma remapping.")
    parser.add_argument("--sigma-max", type=float, default=1.0, help="Upper bound for patch sigma remapping.")
    parser.add_argument(
        "--upper-body-weight",
        type=float,
        default=0.64,
        help="Relative weight assigned to the torso soft mask and clothing identity suppression.",
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


def crop_image(image: np.ndarray, box: BoundingBox) -> np.ndarray | None:
    x1 = int(max(math.floor(box.x1), 0))
    y1 = int(max(math.floor(box.y1), 0))
    x2 = int(min(math.ceil(box.x2), image.shape[1]))
    y2 = int(min(math.ceil(box.y2), image.shape[0]))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def local_boxes_to_global(
    boxes: np.ndarray | tuple[tuple[int, int, int, int], ...],
    person_box: BoundingBox,
    source: str,
) -> list[BoundingBox]:
    global_boxes: list[BoundingBox] = []
    for x, y, w, h in boxes:
        global_boxes.append(
            BoundingBox(
                x1=person_box.x1 + float(x),
                y1=person_box.y1 + float(y),
                x2=person_box.x1 + float(x + w),
                y2=person_box.y1 + float(y + h),
                conf=1.0,
                source=source,
            )
        )
    return global_boxes


def local_boxes_to_global_scaled(
    boxes: np.ndarray | tuple[tuple[int, int, int, int], ...],
    person_box: BoundingBox,
    scale: float,
    source: str,
) -> list[BoundingBox]:
    global_boxes: list[BoundingBox] = []
    for x, y, w, h in boxes:
        global_boxes.append(
            BoundingBox(
                x1=person_box.x1 + float(x) / scale,
                y1=person_box.y1 + float(y) / scale,
                x2=person_box.x1 + float(x + w) / scale,
                y2=person_box.y1 + float(y + h) / scale,
                conf=1.0,
                source=source,
            )
        )
    return global_boxes


def select_best_auxiliary_box(candidates: list[BoundingBox], person_box: BoundingBox) -> BoundingBox | None:
    if not candidates:
        return None

    filtered: list[BoundingBox] = []
    for box in candidates:
        center_y_ratio = (box.center_y - person_box.y1) / max(person_box.height, 1.0)
        width_ratio = box.width / max(person_box.width, 1.0)
        height_ratio = box.height / max(person_box.height, 1.0)
        if center_y_ratio > 0.48:
            continue
        if width_ratio > 0.62 or height_ratio > 0.50:
            continue
        filtered.append(box)

    if filtered:
        candidates = filtered

    person_height = max(person_box.height, 1.0)
    person_area = max(person_box.width * person_box.height, 1.0)

    def score(box: BoundingBox) -> float:
        center_y_ratio = (box.center_y - person_box.y1) / person_height
        top_bonus = max(0.05, 1.35 - center_y_ratio)
        area_ratio = min((box.width * box.height) / person_area, 0.18)
        return box.conf * top_bonus * (0.5 + area_ratio)

    return max(candidates, key=score)


def detect_person_boxes(model: YOLO, image: np.ndarray, conf: float, device: str) -> list[BoundingBox]:
    results = model.predict(source=image, classes=[0], conf=conf, device=device, verbose=False)
    if not results:
        return []

    boxes = results[0].boxes
    if boxes is None:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    detections = [
        BoundingBox(
            x1=float(coords[0]),
            y1=float(coords[1]),
            x2=float(coords[2]),
            y2=float(coords[3]),
            conf=float(confidence),
            source="person_yolo",
        )
        for coords, confidence in zip(xyxy, confs)
    ]
    detections.sort(key=lambda box: (box.y1, box.x1))
    return detections


def expand_box(
    box: BoundingBox,
    scale_x: float,
    scale_y: float,
    image_width: int,
    image_height: int,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    max_width: float | None = None,
    max_height: float | None = None,
    source: str | None = None,
) -> BoundingBox:
    new_width = box.width * scale_x
    new_height = box.height * scale_y
    if max_width is not None:
        new_width = min(new_width, max_width)
    if max_height is not None:
        new_height = min(new_height, max_height)

    center_x = box.center_x + shift_x * box.width
    center_y = box.center_y + shift_y * box.height
    expanded = BoundingBox(
        x1=center_x - new_width / 2.0,
        y1=center_y - new_height / 2.0,
        x2=center_x + new_width / 2.0,
        y2=center_y + new_height / 2.0,
        conf=box.conf,
        source=box.source if source is None else source,
    )
    return expanded.clip(image_width, image_height) or box


def face_box_to_head_components(
    face_box: BoundingBox,
    person_box: BoundingBox,
    image_width: int,
    image_height: int,
) -> tuple[BoundingBox, BoundingBox]:
    head_box = expand_box(
        face_box,
        scale_x=1.62,
        scale_y=2.08,
        shift_y=-0.12,
        max_width=person_box.width * 0.68,
        max_height=person_box.height * 0.52,
        image_width=image_width,
        image_height=image_height,
        source="head_from_face",
    )
    head_core_box = expand_box(
        face_box,
        scale_x=1.34,
        scale_y=1.58,
        shift_y=-0.04,
        max_width=person_box.width * 0.54,
        max_height=person_box.height * 0.38,
        image_width=image_width,
        image_height=image_height,
        source="head_core_from_face",
    )
    return head_box, head_core_box


def head_box_to_core_box(
    head_box: BoundingBox,
    person_box: BoundingBox,
    image_width: int,
    image_height: int,
) -> BoundingBox:
    return expand_box(
        head_box,
        scale_x=0.72,
        scale_y=0.66,
        shift_y=0.06,
        max_width=person_box.width * 0.46,
        max_height=person_box.height * 0.32,
        image_width=image_width,
        image_height=image_height,
        source="head_core_from_head",
    )


def estimate_head_box_from_person(person_box: BoundingBox, image_width: int, image_height: int) -> BoundingBox:
    if person_box.height < 90:
        half_width = 0.35
        top_offset = -0.08
        bottom_offset = 0.54
    elif person_box.height < 160:
        half_width = 0.31
        top_offset = -0.05
        bottom_offset = 0.46
    else:
        half_width = 0.28
        top_offset = -0.03
        bottom_offset = 0.40

    return BoundingBox(
        x1=person_box.center_x - person_box.width * half_width,
        y1=person_box.y1 + person_box.height * top_offset,
        x2=person_box.center_x + person_box.width * half_width,
        y2=person_box.y1 + person_box.height * bottom_offset,
        conf=person_box.conf,
        source="head_from_person",
    ).clip(image_width, image_height) or person_box


def estimate_torso_box(person_box: BoundingBox, head_box: BoundingBox, image_width: int, image_height: int) -> BoundingBox:
    torso_top = max(person_box.y1 + person_box.height * 0.12, head_box.y2 - person_box.height * 0.04)
    torso_bottom = min(person_box.y1 + person_box.height * 0.84, person_box.y2)
    torso_width = person_box.width * 0.88
    torso_box = BoundingBox(
        x1=person_box.center_x - torso_width / 2.0,
        y1=torso_top,
        x2=person_box.center_x + torso_width / 2.0,
        y2=torso_bottom,
        conf=person_box.conf,
        source="torso_from_person",
    )
    return torso_box.clip(image_width, image_height) or person_box


def build_detection_record(
    person_box: BoundingBox,
    image: np.ndarray,
    detector: object | None,
    face_model_kind: str,
) -> DetectionRecord:
    image_height, image_width = image.shape[:2]
    aux_box = detector.detect(image, person_box) if detector is not None else None

    if aux_box is not None:
        aux_kind = "head" if aux_box.source.startswith("yolo_aux_head") else "face"
        if aux_kind == "head":
            head_box = expand_box(
                aux_box,
                scale_x=1.10,
                scale_y=1.14,
                shift_y=-0.03,
                max_width=person_box.width * 0.64,
                max_height=person_box.height * 0.46,
                image_width=image_width,
                image_height=image_height,
                source="head_from_aux_head",
            )
            head_core_box = head_box_to_core_box(head_box, person_box, image_width, image_height)
            head_source = "aux_head_detector"
        else:
            head_box, head_core_box = face_box_to_head_components(aux_box, person_box, image_width, image_height)
            head_source = "face_detector"
    else:
        head_box = estimate_head_box_from_person(person_box, image_width, image_height)
        head_core_box = head_box_to_core_box(head_box, person_box, image_width, image_height)
        head_source = "person_top_prior"

    torso_box = estimate_torso_box(person_box, head_box, image_width, image_height)
    return DetectionRecord(
        person_box=person_box,
        head_box=head_box,
        head_core_box=head_core_box,
        torso_box=torso_box,
        head_source=head_source,
        aux_box=aux_box,
    )


def render_gaussian(
    target: np.ndarray,
    box: BoundingBox,
    amplitude: float,
    sigma_scale_x: float,
    sigma_scale_y: float,
    support_box: BoundingBox | None = None,
) -> None:
    sigma_x = max(box.width * sigma_scale_x, 1.0)
    sigma_y = max(box.height * sigma_scale_y, 1.0)
    radius_x = int(math.ceil(3.0 * sigma_x))
    radius_y = int(math.ceil(3.0 * sigma_y))

    x1 = max(0, int(math.floor(box.center_x - radius_x)))
    y1 = max(0, int(math.floor(box.center_y - radius_y)))
    x2 = min(target.shape[1], int(math.ceil(box.center_x + radius_x)))
    y2 = min(target.shape[0], int(math.ceil(box.center_y + radius_y)))
    if x2 <= x1 or y2 <= y1:
        return

    xs = np.arange(x1, x2, dtype=np.float32)
    ys = np.arange(y1, y2, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    gaussian = amplitude * np.exp(
        -(
            ((grid_x - box.center_x) ** 2) / (2.0 * sigma_x * sigma_x)
            + ((grid_y - box.center_y) ** 2) / (2.0 * sigma_y * sigma_y)
        )
    )

    if support_box is not None:
        support = np.zeros_like(gaussian, dtype=np.float32)
        sx1 = max(x1, int(math.floor(support_box.x1)))
        sy1 = max(y1, int(math.floor(support_box.y1)))
        sx2 = min(x2, int(math.ceil(support_box.x2)))
        sy2 = min(y2, int(math.ceil(support_box.y2)))
        if sx2 > sx1 and sy2 > sy1:
            support[sy1 - y1 : sy2 - y1, sx1 - x1 : sx2 - x1] = 1.0
        gaussian *= support

    target[y1:y2, x1:x2] = np.maximum(target[y1:y2, x1:x2], gaussian)


def render_soft_box(
    target: np.ndarray,
    box: BoundingBox,
    amplitude: float,
    feather_scale_x: float,
    feather_scale_y: float,
    support_box: BoundingBox | None = None,
) -> None:
    feather_x = max(box.width * feather_scale_x, 1.0)
    feather_y = max(box.height * feather_scale_y, 1.0)
    radius_x = int(math.ceil(3.0 * feather_x))
    radius_y = int(math.ceil(3.0 * feather_y))

    x1 = max(0, int(math.floor(box.x1 - radius_x)))
    y1 = max(0, int(math.floor(box.y1 - radius_y)))
    x2 = min(target.shape[1], int(math.ceil(box.x2 + radius_x)))
    y2 = min(target.shape[0], int(math.ceil(box.y2 + radius_y)))
    if x2 <= x1 or y2 <= y1:
        return

    xs = np.arange(x1, x2, dtype=np.float32)
    ys = np.arange(y1, y2, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    dx = np.maximum(np.maximum(box.x1 - grid_x, 0.0), grid_x - box.x2)
    dy = np.maximum(np.maximum(box.y1 - grid_y, 0.0), grid_y - box.y2)
    soft_mask = amplitude * np.exp(-((dx * dx) / (2.0 * feather_x * feather_x) + (dy * dy) / (2.0 * feather_y * feather_y)))

    if support_box is not None:
        support = np.zeros_like(soft_mask, dtype=np.float32)
        sx1 = max(x1, int(math.floor(support_box.x1)))
        sy1 = max(y1, int(math.floor(support_box.y1)))
        sx2 = min(x2, int(math.ceil(support_box.x2)))
        sy2 = min(y2, int(math.ceil(support_box.y2)))
        if sx2 > sx1 and sy2 > sy1:
            support[sy1 - y1 : sy2 - y1, sx1 - x1 : sx2 - x1] = 1.0
        soft_mask *= support

    target[y1:y2, x1:x2] = np.maximum(target[y1:y2, x1:x2], soft_mask)


def build_privacy_map(shape: tuple[int, int], records: list[DetectionRecord], upper_body_weight: float) -> np.ndarray:
    privacy_map = np.zeros(shape, dtype=np.float32)
    image_height, image_width = shape

    for record in records:
        person_map = np.zeros(shape, dtype=np.float32)
        head_support = expand_box(
            record.head_box,
            scale_x=1.22,
            scale_y=1.20,
            shift_y=-0.05,
            image_width=image_width,
            image_height=image_height,
            source="head_support",
        )
        render_soft_box(
            person_map,
            box=record.head_core_box,
            amplitude=1.0,
            feather_scale_x=0.20,
            feather_scale_y=0.18,
            support_box=head_support,
        )
        render_gaussian(
            person_map,
            box=record.head_box,
            amplitude=0.92,
            sigma_scale_x=0.36,
            sigma_scale_y=0.34,
            support_box=head_support,
        )

        torso_support = BoundingBox(
            x1=record.person_box.x1 - record.person_box.width * 0.06,
            y1=record.person_box.y1 + record.person_box.height * 0.10,
            x2=record.person_box.x2 + record.person_box.width * 0.06,
            y2=record.person_box.y1 + record.person_box.height * 0.82,
            source="torso_support",
        ).clip(image_width, image_height)
        render_soft_box(
            person_map,
            box=record.torso_box,
            amplitude=upper_body_weight,
            feather_scale_x=0.22,
            feather_scale_y=0.26,
            support_box=torso_support,
        )
        torso_halo = expand_box(
            record.torso_box,
            scale_x=1.08,
            scale_y=1.12,
            shift_y=0.02,
            image_width=image_width,
            image_height=image_height,
            source="torso_halo",
        )
        render_soft_box(
            person_map,
            box=torso_halo,
            amplitude=upper_body_weight * 0.9,
            feather_scale_x=0.24,
            feather_scale_y=0.30,
            support_box=torso_support,
        )

        privacy_map = np.maximum(privacy_map, person_map)

    return np.clip(privacy_map, 0.0, 1.0)


def aggregate_patch_scores(
    privacy_map: np.ndarray,
    patch_size: int,
    alpha: float,
    sigma_min: float,
    sigma_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = privacy_map.shape
    rows = math.ceil(height / patch_size)
    cols = math.ceil(width / patch_size)
    patch_scores = np.zeros((rows, cols), dtype=np.float32)

    for row in range(rows):
        for col in range(cols):
            y1 = row * patch_size
            x1 = col * patch_size
            patch = privacy_map[y1 : min(y1 + patch_size, height), x1 : min(x1 + patch_size, width)]
            if patch.size == 0:
                continue
            patch_scores[row, col] = alpha * float(patch.max()) + (1.0 - alpha) * float(patch.mean())

    patch_sigmas = np.clip(
        sigma_min + (sigma_max - sigma_min) * patch_scores,
        sigma_min,
        sigma_max,
    ).astype(np.float32)
    return patch_scores, patch_sigmas


def save_heatmap(map_array: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(map_array * 255.0, 0.0, 255.0).astype(np.uint8)
    cv2.imwrite(str(destination), image)


def save_preview_images(
    image: np.ndarray,
    privacy_map: np.ndarray,
    patch_sigmas: np.ndarray,
    sigma_min: float,
    sigma_max: float,
    output_dir: Path,
) -> None:
    heatmap_u8 = np.clip(privacy_map * 255.0, 0.0, 255.0).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(image, 0.52, heatmap_color, 0.48, 0.0)
    cv2.imwrite(str(output_dir / "privacy_overlay.jpg"), overlay)

    blur_strength = max(7, int(round(min(image.shape[:2]) / 24)))
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=blur_strength, sigmaY=blur_strength)
    mask = privacy_map[..., None]
    preview_mask = np.power(mask, 0.72)
    preview_mask = np.where(preview_mask > 0.16, np.maximum(preview_mask, 0.70), preview_mask)
    soft_masked = np.clip(image.astype(np.float32) * (1.0 - preview_mask) + blurred.astype(np.float32) * preview_mask, 0.0, 255.0)
    cv2.imwrite(str(output_dir / "soft_mask_preview.jpg"), soft_masked.astype(np.uint8))

    sigma_map = cv2.resize(patch_sigmas, dsize=(image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    if sigma_max > sigma_min:
        sigma_norm = (sigma_map - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.zeros_like(sigma_map, dtype=np.float32)
    sigma_u8 = np.clip(sigma_norm * 255.0, 0.0, 255.0).astype(np.uint8)
    sigma_color = cv2.applyColorMap(sigma_u8, cv2.COLORMAP_INFERNO)
    sigma_overlay = cv2.addWeighted(image, 0.70, sigma_color, 0.30, 0.0)
    cv2.imwrite(str(output_dir / "patch_sigma_overlay.jpg"), sigma_overlay)


def maybe_resize_for_map(image: np.ndarray, map_size: tuple[int, int] | None) -> tuple[np.ndarray, float, float]:
    if map_size is None:
        return image, 1.0, 1.0
    map_width, map_height = map_size
    resized = cv2.resize(image, (map_width, map_height), interpolation=cv2.INTER_LINEAR)
    scale_x = map_width / image.shape[1]
    scale_y = map_height / image.shape[0]
    return resized, scale_x, scale_y


def write_metadata(
    destination: Path,
    source: Path,
    original_image: np.ndarray,
    map_image: np.ndarray,
    records: list[DetectionRecord],
    args: argparse.Namespace,
    patch_scores: np.ndarray,
    patch_sigmas: np.ndarray,
    uses_opencv_cascade: bool,
    uses_yunet: bool,
    detector_chain: list[str],
) -> None:
    payload = {
        "source": str(source),
        "original_size": {"width": int(original_image.shape[1]), "height": int(original_image.shape[0])},
        "map_size": {"width": int(map_image.shape[1]), "height": int(map_image.shape[0])},
        "patch_size": args.patch_size,
        "patch_alpha": args.patch_alpha,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "upper_body_weight": args.upper_body_weight,
        "person_model": args.person_model,
        "face_model": args.face_model,
        "face_model_kind": args.face_model_kind,
        "yunet_model": str(args.yunet_model),
        "yunet_enabled": uses_yunet,
        "opencv_face_cascade_enabled": uses_opencv_cascade,
        "detector_chain": detector_chain,
        "patch_scores_shape": list(patch_scores.shape),
        "patch_sigmas_shape": list(patch_sigmas.shape),
        "detections": [record.to_dict() for record in records],
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    sources = iter_sources(args.source)
    if not sources:
        raise FileNotFoundError(f"No input files found in {args.source}")

    person_model = YOLO(args.person_model)
    aux_detectors: list[object] = []
    detector_chain: list[str] = []
    uses_opencv_cascade = False
    uses_yunet = False

    if args.face_model:
        aux_detectors.append(YoloAuxDetector(args.face_model, conf=args.face_conf, device=args.device, kind=args.face_model_kind))
        detector_chain.append(f"yolo_aux_{args.face_model_kind}")

    yunet_model_path = args.yunet_model
    if not yunet_model_path.is_absolute():
        yunet_model_path = PROJECT_ROOT / yunet_model_path
    yunet_detector = YuNetFaceDetector(yunet_model_path, score_threshold=args.yunet_score_threshold)
    if yunet_detector.available:
        aux_detectors.append(yunet_detector)
        detector_chain.append("yunet_face")
        uses_yunet = True

    opencv_detector = OpenCvFaceDetector()
    if opencv_detector.available:
        aux_detectors.append(opencv_detector)
        detector_chain.append("opencv_face_cascade")
        uses_opencv_cascade = True

    aux_detector = FallbackFaceDetector(aux_detectors) if aux_detectors else None

    args.output.mkdir(parents=True, exist_ok=True)

    for src in sources:
        image = cv2.imread(str(src))
        if image is None:
            raise FileNotFoundError(f"Failed to read image {src}")

        person_boxes = detect_person_boxes(person_model, image, conf=args.person_conf, device=args.device)
        records = [build_detection_record(box, image, aux_detector, args.face_model_kind) for box in person_boxes]

        map_image, scale_x, scale_y = maybe_resize_for_map(image, tuple(args.map_size) if args.map_size else None)
        scaled_records = [
            DetectionRecord(
                person_box=record.person_box.scale(scale_x, scale_y),
                head_box=record.head_box.scale(scale_x, scale_y),
                head_core_box=record.head_core_box.scale(scale_x, scale_y),
                torso_box=record.torso_box.scale(scale_x, scale_y),
                head_source=record.head_source,
                aux_box=None if record.aux_box is None else record.aux_box.scale(scale_x, scale_y),
            )
            for record in records
        ]

        privacy_map = build_privacy_map(map_image.shape[:2], scaled_records, upper_body_weight=args.upper_body_weight)
        patch_scores, patch_sigmas = aggregate_patch_scores(
            privacy_map,
            patch_size=args.patch_size,
            alpha=args.patch_alpha,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
        )

        output_dir = args.output / src.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        save_heatmap(privacy_map, output_dir / "privacy_map.png")
        np.save(output_dir / "privacy_map.npy", privacy_map)
        np.save(output_dir / "patch_scores.npy", patch_scores)
        np.save(output_dir / "patch_sigmas.npy", patch_sigmas)
        save_preview_images(map_image, privacy_map, patch_sigmas, args.sigma_min, args.sigma_max, output_dir)
        write_metadata(
            output_dir / "metadata.json",
            source=src,
            original_image=image,
            map_image=map_image,
            records=scaled_records,
            args=args,
            patch_scores=patch_scores,
            patch_sigmas=patch_sigmas,
            uses_opencv_cascade=uses_opencv_cascade,
            uses_yunet=uses_yunet,
            detector_chain=detector_chain,
        )

        print(
            f"{src}: detected {len(records)} person(s), "
            f"privacy map {privacy_map.shape[1]}x{privacy_map.shape[0]}, "
            f"patch grid {patch_sigmas.shape[1]}x{patch_sigmas.shape[0]}, "
            f"saved to {output_dir}"
        )


if __name__ == "__main__":
    main()


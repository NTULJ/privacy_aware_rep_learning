from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor
from ultralytics import YOLO

import generate_privacy_prior as prior
import qwen3_vl_firstlayer_dp as dp


DEFAULT_STAGES = (
    "x_pre_clean",
    "x_pre_priv",
    "block1_clean",
    "block1_priv",
    "block16_clean",
    "block16_priv",
    "hpool_clean",
    "hpool_priv",
)
DEFAULT_POOLINGS = ("global_mean", "head_mean", "torso_mean", "nonhead_human_mean")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BLOCK_STAGE_TO_INDEX = {
    "block1": 0,
    "block8": 8,
    "block16": 16,
    "block24": 24,
}


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    image_path: str
    split: str
    label: str
    dataset: str
    person_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract clean/noisy Qwen3-VL vision features and patch metadata for privacy-utility evaluation."
    )
    parser.add_argument("--dataset-root", type=Path, default=None, help="Root directory containing input images.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional JSONL/CSV manifest. Overrides dataset scanning.")
    parser.add_argument(
        "--dataset-name",
        choices=("auto", "stanford40", "lfw", "generic"),
        default="auto",
        help="Dataset layout used when inferring labels from --dataset-root.",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct", help="Qwen3-VL model id or local path.")
    parser.add_argument("--mode", choices=("vision-only", "full-model"), default="vision-only")
    parser.add_argument("--output", type=Path, required=True, help="Output experiment directory.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--delta-priv", type=float, required=True)
    parser.add_argument("--delta-mask", type=float, default=1e-3)
    parser.add_argument("--noise-scale-multiplier", type=float, default=1.0)
    parser.add_argument("--clip-norm", type=float, default=32.0)
    parser.add_argument("--patch-alpha", type=float, default=0.70)
    parser.add_argument("--upper-body-weight", type=float, default=0.64)
    parser.add_argument("--batch-size", type=int, default=1, help="Currently only batch_size=1 is supported.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-token-features", action="store_true", help="Save per-sample token tensors to tokens/<sample_id>.npz.")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=list(DEFAULT_STAGES),
        help="Stage keys to export. Supported: x_pre_clean/priv, block1/8/16/24_clean/priv, hpool_clean/priv.",
    )
    parser.add_argument(
        "--poolings",
        nargs="+",
        default=list(DEFAULT_POOLINGS),
        help="Pooling strategies for pooled/<stage>__<pooling>.npy.",
    )
    parser.add_argument("--person-model", default="yolo11n.pt")
    parser.add_argument("--person-conf", type=float, default=0.25)
    parser.add_argument("--face-model", default=None)
    parser.add_argument("--face-model-kind", choices=("face", "head"), default="face")
    parser.add_argument("--face-conf", type=float, default=0.20)
    parser.add_argument("--yunet-model", type=Path, default=Path("models/face_detection_yunet_2023mar.onnx"))
    parser.add_argument("--yunet-score-threshold", type=float, default=0.55)
    return parser.parse_args()


def ensure_args_valid(args: argparse.Namespace) -> None:
    if args.batch_size != 1:
        raise ValueError("This first extractor version supports only --batch-size 1.")
    if args.manifest is None and args.dataset_root is None:
        raise ValueError("Either --manifest or --dataset-root must be provided.")
    unsupported = [stage for stage in args.stages if stage not in supported_stage_names()]
    if unsupported:
        raise ValueError(f"Unsupported stages: {unsupported}")
    unsupported_poolings = [pooling for pooling in args.poolings if pooling not in DEFAULT_POOLINGS]
    if unsupported_poolings:
        raise ValueError(f"Unsupported poolings: {unsupported_poolings}")
    if args.mode == "full-model":
        raise NotImplementedError("extract_qwen_vl_features.py currently supports --mode vision-only only.")


def supported_stage_names() -> set[str]:
    names = {"x_pre_clean", "x_pre_priv", "hpool_clean", "hpool_priv"}
    for block_name in BLOCK_STAGE_TO_INDEX:
        names.add(f"{block_name}_clean")
        names.add(f"{block_name}_priv")
    return names


def slugify_sample_id(raw: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.replace("\\", "/"))
    return slug.strip("._") or "sample"


def infer_flat_label(stem: str) -> str:
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def infer_record_from_path(image_path: Path, dataset_root: Path, dataset_name: str) -> SampleRecord:
    relative = image_path.relative_to(dataset_root)
    sample_id = slugify_sample_id(str(relative.with_suffix("")))

    if dataset_name == "lfw":
        label = image_path.parent.name
        person_id = label
        dataset = "lfw"
    elif dataset_name == "stanford40":
        label = infer_flat_label(image_path.stem)
        person_id = ""
        dataset = "stanford40"
    elif dataset_name == "generic":
        label = image_path.parent.name if image_path.parent != dataset_root else infer_flat_label(image_path.stem)
        person_id = label if image_path.parent != dataset_root else ""
        dataset = dataset_root.name
    else:
        if image_path.parent != dataset_root:
            label = image_path.parent.name
            person_id = label
            dataset = dataset_root.name.lower()
        else:
            label = infer_flat_label(image_path.stem)
            person_id = ""
            dataset = "stanford40" if dataset_root.name.lower() == "images" else dataset_root.name.lower()

    return SampleRecord(
        sample_id=sample_id,
        image_path=str(image_path.resolve()),
        split="all",
        label=label,
        dataset=dataset,
        person_id=person_id,
    )


def scan_dataset(dataset_root: Path, dataset_name: str) -> list[SampleRecord]:
    files = sorted(path for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    return [infer_record_from_path(path, dataset_root, dataset_name) for path in files]


def load_manifest(manifest_path: Path, dataset_root: Path | None) -> list[SampleRecord]:
    if manifest_path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif manifest_path.suffix.lower() == ".csv":
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError(f"Unsupported manifest format: {manifest_path}")

    samples: list[SampleRecord] = []
    for index, row in enumerate(rows):
        raw_path = Path(row["image_path"])
        if not raw_path.is_absolute():
            if dataset_root is not None:
                raw_path = dataset_root / raw_path
            else:
                raw_path = manifest_path.parent / raw_path
        sample_id = row.get("sample_id") or slugify_sample_id(f"{index:06d}_{raw_path.stem}")
        samples.append(
            SampleRecord(
                sample_id=sample_id,
                image_path=str(raw_path.resolve()),
                split=row.get("split", "all"),
                label=row.get("label", infer_flat_label(raw_path.stem)),
                dataset=row.get("dataset", raw_path.parent.name),
                person_id=row.get("person_id", ""),
            )
        )
    return samples


def unique_samples(samples: list[SampleRecord]) -> list[SampleRecord]:
    seen: dict[str, int] = {}
    deduped: list[SampleRecord] = []
    for sample in samples:
        count = seen.get(sample.sample_id, 0)
        seen[sample.sample_id] = count + 1
        if count == 0:
            deduped.append(sample)
            continue
        deduped.append(
            SampleRecord(
                sample_id=f"{sample.sample_id}_{count}",
                image_path=sample.image_path,
                split=sample.split,
                label=sample.label,
                dataset=sample.dataset,
                person_id=sample.person_id,
            )
        )
    return deduped


def parse_samples(args: argparse.Namespace) -> list[SampleRecord]:
    if args.manifest is not None:
        samples = load_manifest(args.manifest, dataset_root=args.dataset_root)
    else:
        samples = scan_dataset(args.dataset_root.resolve(), args.dataset_name)
    samples = unique_samples(samples)
    if args.limit is not None:
        samples = samples[: args.limit]
    return samples


def build_runtime_detectors(args: argparse.Namespace) -> tuple[YOLO, object | None]:
    person_model = YOLO(args.person_model)
    aux_detector, _, _ = dp.build_aux_detector(args)
    return person_model, aux_detector


def build_privacy_context(
    image_bgr: np.ndarray,
    person_model: YOLO,
    aux_detector: object | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[prior.DetectionRecord]]:
    person_boxes = prior.detect_person_boxes(person_model, image_bgr, conf=args.person_conf, device=args.device)
    records = [prior.build_detection_record(box, image_bgr, aux_detector, args.face_model_kind) for box in person_boxes]
    privacy_map = prior.build_privacy_map(image_bgr.shape[:2], records, upper_body_weight=args.upper_body_weight)
    return privacy_map, records


def rasterize_boxes(
    records: list[prior.DetectionRecord],
    original_shape: tuple[int, int],
    resized_width: int,
    resized_height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    original_height, original_width = original_shape
    scale_x = resized_width / float(original_width)
    scale_y = resized_height / float(original_height)

    head_mask = np.zeros((resized_height, resized_width), dtype=np.uint8)
    torso_mask = np.zeros_like(head_mask)
    human_mask = np.zeros_like(head_mask)

    def fill_box(target: np.ndarray, box: prior.BoundingBox) -> None:
        scaled = box.scale(scale_x, scale_y).clip(resized_width, resized_height)
        if scaled is None:
            return
        x1 = max(0, int(np.floor(scaled.x1)))
        y1 = max(0, int(np.floor(scaled.y1)))
        x2 = min(resized_width, int(np.ceil(scaled.x2)))
        y2 = min(resized_height, int(np.ceil(scaled.y2)))
        if x2 > x1 and y2 > y1:
            target[y1:y2, x1:x2] = 1

    for record in records:
        fill_box(head_mask, record.head_box)
        fill_box(torso_mask, record.torso_box)
        fill_box(human_mask, record.person_box)

    return head_mask, torso_mask, human_mask


def aggregate_binary_mask(mask: np.ndarray, patch_size: int) -> np.ndarray:
    height, width = mask.shape
    rows = height // patch_size
    cols = width // patch_size
    reshaped = mask.reshape(rows, patch_size, cols, patch_size)
    return reshaped.max(axis=(1, 3)).astype(bool)


def merge_patch_mask(mask: np.ndarray, merge_size: int) -> np.ndarray:
    if merge_size == 1:
        return mask.astype(bool)
    rows, cols = mask.shape
    if rows % merge_size != 0 or cols % merge_size != 0:
        raise ValueError(f"Patch mask shape {mask.shape} is not divisible by merge size {merge_size}.")
    reshaped = mask.reshape(rows // merge_size, merge_size, cols // merge_size, merge_size)
    return reshaped.max(axis=(1, 3)).astype(bool)


def merge_patch_score_grid(scores: np.ndarray, merge_size: int) -> np.ndarray:
    if merge_size == 1:
        return scores.astype(np.float32)
    rows, cols = scores.shape
    reshaped = scores.reshape(rows // merge_size, merge_size, cols // merge_size, merge_size)
    return reshaped.max(axis=(1, 3)).astype(np.float32)


def flatten_mask(mask: np.ndarray) -> np.ndarray:
    return mask.astype(bool).reshape(-1)


def masked_mean(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.any():
        return features[mask].mean(axis=0, dtype=np.float32)
    return features.mean(axis=0, dtype=np.float32)


def pooling_masks_for_stage(
    stage_name: str,
    head_mask: np.ndarray,
    torso_mask: np.ndarray,
    human_mask: np.ndarray,
    head_mask_merged: np.ndarray,
    torso_mask_merged: np.ndarray,
    human_mask_merged: np.ndarray,
) -> dict[str, np.ndarray]:
    if stage_name.startswith("hpool"):
        head = flatten_mask(head_mask_merged)
        torso = flatten_mask(torso_mask_merged)
        human = flatten_mask(human_mask_merged)
    else:
        head = flatten_mask(head_mask)
        torso = flatten_mask(torso_mask)
        human = flatten_mask(human_mask)

    nonhead_human = human & ~head
    return {
        "global_mean": np.ones_like(human, dtype=bool),
        "head_mean": head,
        "torso_mean": torso,
        "nonhead_human_mean": nonhead_human,
    }


def build_position_embeddings(visual_model: torch.nn.Module, grid_thw: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    rotary_pos_emb = visual_model.rot_pos_emb(grid_thw)
    seq_len = rotary_pos_emb.shape[0]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
    return position_embeddings, cu_seqlens


def forward_blocks_collect(
    visual_model: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    cu_seqlens: torch.Tensor,
    capture_block_indices: set[int],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    captured: dict[str, np.ndarray] = {}
    for layer_index, block in enumerate(visual_model.blocks):
        hidden_states = block(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
        if layer_index in capture_block_indices:
            captured[f"block{layer_index}"] = hidden_states.detach().cpu().float().numpy()
    merged_hidden_states = visual_model.merger(hidden_states)
    return captured, merged_hidden_states.detach().cpu().float().numpy()


def requested_block_indices(stages: list[str]) -> set[int]:
    indices: set[int] = set()
    for stage in stages:
        base = stage.rsplit("_", 1)[0]
        if base in BLOCK_STAGE_TO_INDEX:
            indices.add(BLOCK_STAGE_TO_INDEX[base])
    return indices


def collect_stage_features(
    visual_model: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    row_noise_scales: torch.Tensor,
    clip_norm: float,
    stages: list[str],
) -> tuple[dict[str, np.ndarray], dp.ClipStats]:
    hidden_states = visual_model.patch_embed(pixel_values)
    clean_x_pre = hidden_states

    noisy_x_pre, clip_stats = dp.apply_patchwise_dp_noise(
        hidden_states=clean_x_pre,
        row_noise_scales=row_noise_scales.to(device=clean_x_pre.device, dtype=clean_x_pre.dtype),
        clip_norm=clip_norm,
    )

    pos_embeds = visual_model.fast_pos_embed_interpolate(grid_thw)
    clean_block_input = clean_x_pre + pos_embeds
    noisy_block_input = noisy_x_pre + pos_embeds

    position_embeddings, cu_seqlens = build_position_embeddings(visual_model, grid_thw)
    capture_indices = requested_block_indices(stages)

    clean_block_outputs, clean_hpool = forward_blocks_collect(
        visual_model=visual_model,
        hidden_states=clean_block_input,
        position_embeddings=position_embeddings,
        cu_seqlens=cu_seqlens,
        capture_block_indices=capture_indices,
    )
    noisy_block_outputs, noisy_hpool = forward_blocks_collect(
        visual_model=visual_model,
        hidden_states=noisy_block_input,
        position_embeddings=position_embeddings,
        cu_seqlens=cu_seqlens,
        capture_block_indices=capture_indices,
    )

    features: dict[str, np.ndarray] = {}
    if "x_pre_clean" in stages:
        features["x_pre_clean"] = clean_x_pre.detach().cpu().float().numpy()
    if "x_pre_priv" in stages:
        features["x_pre_priv"] = noisy_x_pre.detach().cpu().float().numpy()
    if "hpool_clean" in stages:
        features["hpool_clean"] = clean_hpool
    if "hpool_priv" in stages:
        features["hpool_priv"] = noisy_hpool

    for block_name, layer_index in BLOCK_STAGE_TO_INDEX.items():
        clean_key = f"{block_name}_clean"
        priv_key = f"{block_name}_priv"
        tensor_key = f"block{layer_index}"
        if clean_key in stages:
            features[clean_key] = clean_block_outputs[tensor_key]
        if priv_key in stages:
            features[priv_key] = noisy_block_outputs[tensor_key]
    return features, clip_stats


def save_jsonl(destination: Path, records: list[dict[str, object]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_csv(destination: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    ensure_args_valid(args)

    samples = parse_samples(args)
    if not samples:
        raise RuntimeError("No input samples were found.")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "patch").mkdir(parents=True, exist_ok=True)
    if args.save_token_features:
        (args.output / "tokens").mkdir(parents=True, exist_ok=True)
    (args.output / "pooled").mkdir(parents=True, exist_ok=True)
    (args.output / "stats").mkdir(parents=True, exist_ok=True)

    source_id = dp.resolve_pretrained_source(args.model_id, args.local_files_only)
    hf_kwargs = {"local_files_only": args.local_files_only}
    image_processor = AutoImageProcessor.from_pretrained(source_id, **hf_kwargs)
    config = AutoConfig.from_pretrained(source_id, **hf_kwargs)
    dtype = dp.dtype_from_name(args.dtype, mode=args.mode)
    device = torch.device(args.device)
    visual_model = dp.load_visual_model_from_checkpoint(Path(source_id), config, device=device, dtype=dtype)

    person_model, aux_detector = build_runtime_detectors(args)
    stages = list(args.stages)
    poolings = list(args.poolings)

    pooled_buffers: dict[str, list[np.ndarray]] = {f"{stage}__{pooling}": [] for stage in stages for pooling in poolings}
    manifest_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for offset, sample in enumerate(samples, start=1):
        try:
            image_path = Path(sample.image_path)
            image_pil = Image.open(image_path).convert("RGB")
            image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            image_inputs = image_processor(images=image_pil, return_tensors="pt")
            pixel_values = image_inputs["pixel_values"]
            image_grid_thw = image_inputs["image_grid_thw"]

            privacy_map, records = build_privacy_context(
                image_bgr=image_bgr,
                person_model=person_model,
                aux_detector=aux_detector,
                args=args,
            )
            resized_map, patch_scores_np = dp.aggregate_scores_for_qwen_grid(
                privacy_map=privacy_map,
                grid_thw=image_grid_thw,
                vision_patch_size=int(config.vision_config.patch_size),
                patch_alpha=args.patch_alpha,
            )
            patch_scores = torch.from_numpy(patch_scores_np.reshape(-1))
            patch_noise, calibration = dp.build_patch_noise_factors(
                patch_scores=patch_scores,
                epsilon=args.epsilon,
                delta_priv=args.delta_priv,
                delta_mask=args.delta_mask,
                clip_norm=args.clip_norm,
                noise_scale_multiplier=args.noise_scale_multiplier,
            )

            resized_height, resized_width = resized_map.shape
            head_mask_px, torso_mask_px, human_mask_px = rasterize_boxes(
                records=records,
                original_shape=image_bgr.shape[:2],
                resized_width=resized_width,
                resized_height=resized_height,
            )
            patch_size = int(config.vision_config.patch_size)
            head_patch_mask = aggregate_binary_mask(head_mask_px, patch_size)
            torso_patch_mask = aggregate_binary_mask(torso_mask_px, patch_size)
            human_patch_mask = aggregate_binary_mask(human_mask_px, patch_size)
            background_patch_mask = ~human_patch_mask

            merge_size = int(config.vision_config.spatial_merge_size)
            head_patch_mask_merged = merge_patch_mask(head_patch_mask, merge_size)
            torso_patch_mask_merged = merge_patch_mask(torso_patch_mask, merge_size)
            human_patch_mask_merged = merge_patch_mask(human_patch_mask, merge_size)
            background_patch_mask_merged = ~human_patch_mask_merged
            merged_patch_scores = merge_patch_score_grid(patch_scores_np, merge_size)

            with torch.inference_mode():
                features, clip_stats = collect_stage_features(
                    visual_model=visual_model,
                    pixel_values=pixel_values.to(device=device, dtype=visual_model.dtype),
                    grid_thw=image_grid_thw.to(device=device),
                    row_noise_scales=patch_noise.row_noise_scales,
                    clip_norm=args.clip_norm,
                    stages=stages,
                )

            for stage_name, stage_features in features.items():
                masks = pooling_masks_for_stage(
                    stage_name=stage_name,
                    head_mask=head_patch_mask,
                    torso_mask=torso_patch_mask,
                    human_mask=human_patch_mask,
                    head_mask_merged=head_patch_mask_merged,
                    torso_mask_merged=torso_patch_mask_merged,
                    human_mask_merged=human_patch_mask_merged,
                )
                for pooling_name in poolings:
                    pooled_buffers[f"{stage_name}__{pooling_name}"].append(
                        masked_mean(stage_features, masks[pooling_name]).astype(np.float32)
                    )

            token_path = ""
            if args.save_token_features:
                token_path = f"tokens/{sample.sample_id}.npz"
                np.savez_compressed(args.output / token_path, **features)

            patch_path = f"patch/{sample.sample_id}.npz"
            np.savez_compressed(
                args.output / patch_path,
                patch_scores=patch_scores_np.astype(np.float32),
                patch_scores_merged=merged_patch_scores.astype(np.float32),
                patch_cov_diag=patch_noise.covariance_diag.reshape(patch_scores_np.shape).cpu().numpy().astype(np.float32),
                row_noise_scales=patch_noise.row_noise_scales.reshape(patch_scores_np.shape).cpu().numpy().astype(np.float32),
                head_patch_mask=head_patch_mask.astype(np.bool_),
                torso_patch_mask=torso_patch_mask.astype(np.bool_),
                human_patch_mask=human_patch_mask.astype(np.bool_),
                background_patch_mask=background_patch_mask.astype(np.bool_),
                head_patch_mask_merged=head_patch_mask_merged.astype(np.bool_),
                torso_patch_mask_merged=torso_patch_mask_merged.astype(np.bool_),
                human_patch_mask_merged=human_patch_mask_merged.astype(np.bool_),
                background_patch_mask_merged=background_patch_mask_merged.astype(np.bool_),
                grid_thw=image_grid_thw.cpu().numpy().astype(np.int32),
                merge_size=np.array([merge_size], dtype=np.int32),
                clip_norm=np.array([args.clip_norm], dtype=np.float32),
                clip_scale=np.array([clip_stats.clip_scale], dtype=np.float32),
                original_fro_norm=np.array([clip_stats.original_fro_norm], dtype=np.float32),
                clipped_fro_norm=np.array([clip_stats.clipped_fro_norm], dtype=np.float32),
                calibration_base_noise_std=np.array([calibration.base_noise_std], dtype=np.float32),
                calibration_upper_bound_scalar=np.array([calibration.upper_bound_scalar], dtype=np.float32),
            )

            manifest_rows.append(asdict(sample))
            label_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "label": sample.label,
                    "split": sample.split,
                    "dataset": sample.dataset,
                    "person_id": sample.person_id,
                    "image_path": sample.image_path,
                }
            )
            index_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "image_path": sample.image_path,
                    "split": sample.split,
                    "label": sample.label,
                    "dataset": sample.dataset,
                    "person_id": sample.person_id,
                    "token_path": token_path,
                    "patch_path": patch_path,
                }
            )
            print(f"[{offset}/{len(samples)}] extracted {sample.sample_id}")
        except Exception as exc:
            failures.append({"sample_id": sample.sample_id, "image_path": sample.image_path, "error": repr(exc)})
            print(f"[{offset}/{len(samples)}] failed {sample.sample_id}: {exc}")

    for feature_key, vectors in pooled_buffers.items():
        if not vectors:
            continue
        np.save(args.output / "pooled" / f"{feature_key}.npy", np.stack(vectors).astype(np.float32))

    save_jsonl(args.output / "manifest.jsonl", manifest_rows)
    write_csv(
        args.output / "labels.csv",
        label_rows,
        fieldnames=["sample_id", "label", "split", "dataset", "person_id", "image_path"],
    )
    write_csv(
        args.output / "index.csv",
        index_rows,
        fieldnames=["sample_id", "image_path", "split", "label", "dataset", "person_id", "token_path", "patch_path"],
    )
    (args.output / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    (args.output / "stats" / "extraction_summary.json").write_text(
        json.dumps(
            {
                "requested_samples": len(samples),
                "successful_samples": len(index_rows),
                "failed_samples": len(failures),
                "stages": stages,
                "poolings": poolings,
                "token_features_saved": args.save_token_features,
                "failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"saved features to {args.output} | "
        f"success={len(index_rows)} failed={len(failures)} "
        f"stages={stages}"
    )


if __name__ == "__main__":
    main()


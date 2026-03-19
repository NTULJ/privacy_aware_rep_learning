import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor

import extract_qwen_vl_features as base
import qwen3_vl_firstlayer_dp as dp


INJECTION_STAGE_TO_BLOCK_INDEX = {
    "block1": 0,
    "block8": 8,
    "block16": 16,
    "block24": 24,
}
DEFAULT_STAGES = (
    "x_pre_clean",
    "x_pre_priv",
    "block1_clean",
    "block1_priv",
    "block8_clean",
    "block8_priv",
    "block16_clean",
    "block16_priv",
    "block24_clean",
    "block24_priv",
    "hpool_clean",
    "hpool_priv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Qwen3-VL features with DP noise injected at a configurable stage."
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dataset-name", choices=("auto", "stanford40", "lfw", "generic"), default="auto")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--mode", choices=("vision-only", "full-model"), default="vision-only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--delta-priv", type=float, required=True)
    parser.add_argument("--delta-mask", type=float, default=1e-3)
    parser.add_argument(
        "--injection-stage",
        choices=("x_pre", "block1", "block8", "block16", "block24", "hpool"),
        default="x_pre",
    )
    parser.add_argument("--noise-scale-multiplier", type=float, default=1.0)
    parser.add_argument("--clip-norm", type=float, default=32.0)
    parser.add_argument("--patch-alpha", type=float, default=0.70)
    parser.add_argument("--upper-body-weight", type=float, default=0.64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-token-features", action="store_true")
    parser.add_argument("--stages", nargs="+", default=list(DEFAULT_STAGES))
    parser.add_argument("--poolings", nargs="+", default=list(base.DEFAULT_POOLINGS))
    parser.add_argument("--person-model", default="yolo11n.pt")
    parser.add_argument("--person-conf", type=float, default=0.25)
    parser.add_argument("--face-model", default=None)
    parser.add_argument("--face-model-kind", choices=("face", "head"), default="face")
    parser.add_argument("--face-conf", type=float, default=0.20)
    parser.add_argument("--yunet-model", type=Path, default=Path("models/face_detection_yunet_2023mar.onnx"))
    parser.add_argument("--yunet-score-threshold", type=float, default=0.55)
    return parser.parse_args()


def collect_stage_features(
    visual_model: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    row_noise_scales: torch.Tensor,
    clip_norm: float,
    stages: list[str],
    injection_stage: str,
) -> tuple[dict[str, np.ndarray], dp.ClipStats]:
    clean_x_pre = visual_model.patch_embed(pixel_values)
    row_noise_scales = row_noise_scales.to(device=clean_x_pre.device, dtype=clean_x_pre.dtype)

    pos_embeds = visual_model.fast_pos_embed_interpolate(grid_thw)
    clean_block_input = clean_x_pre + pos_embeds
    position_embeddings, cu_seqlens = base.build_position_embeddings(visual_model, grid_thw)
    capture_indices = base.requested_block_indices(stages)

    clean_block_outputs, clean_hpool = base.forward_blocks_collect(
        visual_model=visual_model,
        hidden_states=clean_block_input,
        position_embeddings=position_embeddings,
        cu_seqlens=cu_seqlens,
        capture_block_indices=capture_indices,
    )

    clip_stats: dp.ClipStats | None = None
    if injection_stage == "x_pre":
        noisy_x_pre, clip_stats = dp.apply_patchwise_dp_noise(
            hidden_states=clean_x_pre,
            row_noise_scales=row_noise_scales,
            clip_norm=clip_norm,
        )
    else:
        noisy_x_pre = clean_x_pre

    noisy_hidden_states = noisy_x_pre + pos_embeds
    noisy_block_outputs: dict[str, np.ndarray] = {}
    for layer_index, block in enumerate(visual_model.blocks):
        noisy_hidden_states = block(
            noisy_hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        if INJECTION_STAGE_TO_BLOCK_INDEX.get(injection_stage) == layer_index:
            noisy_hidden_states, clip_stats = dp.apply_patchwise_dp_noise(
                hidden_states=noisy_hidden_states,
                row_noise_scales=row_noise_scales,
                clip_norm=clip_norm,
            )
        if layer_index in capture_indices:
            noisy_block_outputs[f"block{layer_index}"] = noisy_hidden_states.detach().cpu().float().numpy()

    if injection_stage == "hpool":
        noisy_hidden_states, clip_stats = dp.apply_patchwise_dp_noise(
            hidden_states=noisy_hidden_states,
            row_noise_scales=row_noise_scales,
            clip_norm=clip_norm,
        )
    if clip_stats is None:
        raise RuntimeError(f"Injection stage {injection_stage} did not produce clip stats.")
    noisy_hpool = visual_model.merger(noisy_hidden_states).detach().cpu().float().numpy()

    features: dict[str, np.ndarray] = {}
    if "x_pre_clean" in stages:
        features["x_pre_clean"] = clean_x_pre.detach().cpu().float().numpy()
    if "x_pre_priv" in stages:
        features["x_pre_priv"] = noisy_x_pre.detach().cpu().float().numpy()
    if "hpool_clean" in stages:
        features["hpool_clean"] = clean_hpool
    if "hpool_priv" in stages:
        features["hpool_priv"] = noisy_hpool

    for block_name, layer_index in base.BLOCK_STAGE_TO_INDEX.items():
        clean_key = f"{block_name}_clean"
        priv_key = f"{block_name}_priv"
        tensor_key = f"block{layer_index}"
        if clean_key in stages:
            features[clean_key] = clean_block_outputs[tensor_key]
        if priv_key in stages:
            features[priv_key] = noisy_block_outputs[tensor_key]
    return features, clip_stats


def main() -> None:
    args = parse_args()
    base.ensure_args_valid(args)

    samples = base.parse_samples(args)
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

    person_model, aux_detector = base.build_runtime_detectors(args)
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

            privacy_map, records = base.build_privacy_context(
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
            head_mask_px, torso_mask_px, human_mask_px = base.rasterize_boxes(
                records=records,
                original_shape=image_bgr.shape[:2],
                resized_width=resized_width,
                resized_height=resized_height,
            )
            patch_size = int(config.vision_config.patch_size)
            head_patch_mask = base.aggregate_binary_mask(head_mask_px, patch_size)
            torso_patch_mask = base.aggregate_binary_mask(torso_mask_px, patch_size)
            human_patch_mask = base.aggregate_binary_mask(human_mask_px, patch_size)
            background_patch_mask = ~human_patch_mask

            merge_size = int(config.vision_config.spatial_merge_size)
            head_patch_mask_merged = base.merge_patch_mask(head_patch_mask, merge_size)
            torso_patch_mask_merged = base.merge_patch_mask(torso_patch_mask, merge_size)
            human_patch_mask_merged = base.merge_patch_mask(human_patch_mask, merge_size)
            background_patch_mask_merged = ~human_patch_mask_merged
            merged_patch_scores = base.merge_patch_score_grid(patch_scores_np, merge_size)

            with torch.inference_mode():
                features, clip_stats = collect_stage_features(
                    visual_model=visual_model,
                    pixel_values=pixel_values.to(device=device, dtype=visual_model.dtype),
                    grid_thw=image_grid_thw.to(device=device),
                    row_noise_scales=patch_noise.row_noise_scales,
                    clip_norm=args.clip_norm,
                    stages=stages,
                    injection_stage=args.injection_stage,
                )

            for stage_name, stage_features in features.items():
                masks = base.pooling_masks_for_stage(
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
                        base.masked_mean(stage_features, masks[pooling_name]).astype(np.float32)
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
        if vectors:
            np.save(args.output / "pooled" / f"{feature_key}.npy", np.stack(vectors).astype(np.float32))

    base.save_jsonl(args.output / "manifest.jsonl", manifest_rows)
    base.write_csv(
        args.output / "labels.csv",
        label_rows,
        fieldnames=["sample_id", "label", "split", "dataset", "person_id", "image_path"],
    )
    base.write_csv(
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
                "injection_stage": args.injection_stage,
                "failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"saved features to {args.output} | success={len(index_rows)} failed={len(failures)} "
        f"injection_stage={args.injection_stage} stages={stages}"
    )


if __name__ == "__main__":
    main()

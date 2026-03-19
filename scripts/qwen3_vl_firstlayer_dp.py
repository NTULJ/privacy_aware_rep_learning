from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for env_name, relative_path in {
    "YOLO_CONFIG_DIR": ".ultralytics",
    "MPLCONFIGDIR": ".matplotlib",
    "TORCH_HOME": ".torch",
    "HF_HOME": ".hf_home",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
}.items():
    if env_name == "HF_HUB_DISABLE_SYMLINKS_WARNING":
        os.environ.setdefault(env_name, relative_path)
        continue
    cache_dir = PROJECT_ROOT / relative_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(env_name, str(cache_dir))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoImageProcessor, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures, Qwen3VLVisionModel
from ultralytics import YOLO

import generate_privacy_prior as prior


@dataclass(frozen=True)
class AnalyticGaussianCalibration:
    epsilon: float
    delta_priv: float
    sensitivity: float
    upper_bound_scalar: float
    base_noise_std: float
    delta_threshold: float


@dataclass(frozen=True)
class ClipStats:
    clip_norm: float
    original_fro_norm: float
    clipped_fro_norm: float
    clip_scale: float


@dataclass(frozen=True)
class PatchNoiseSummary:
    num_patches: int
    grid_t: int
    grid_h: int
    grid_w: int
    delta_mask: float
    noise_scale_multiplier: float
    required_min_singular_value: float
    left_factor_scale: float
    trace_normalized_sum: float
    min_patch_score: float
    max_patch_score: float
    min_cov_diag: float
    max_cov_diag: float
    min_row_scale: float
    max_row_scale: float


@dataclass(frozen=True)
class DPForwardArtifacts:
    calibration: AnalyticGaussianCalibration
    clip_stats: ClipStats
    patch_noise: PatchNoiseSummary
    visual_hidden_size: int
    model_id: str
    mode: str


@dataclass(frozen=True)
class PatchNoiseFactors:
    covariance_diag: torch.Tensor
    row_noise_scales: torch.Tensor
    required_min_singular_value: float
    left_factor_scale: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject patch-aware DP-Forward noise into Qwen3-VL vision tokens before the first vision block."
    )
    parser.add_argument("--image", type=Path, required=True, help="Input image path.")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Qwen3-VL model id or local path.",
    )
    parser.add_argument(
        "--mode",
        choices=("vision-only", "full-model"),
        default="vision-only",
        help="Load only the vision tower weights or attempt to load the full Qwen3-VL model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/qwen3_vl_dp"),
        help="Output directory.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu, cuda:0.")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Skip loading pretrained weights and validate only the DP path math.")
    parser.add_argument("--local-files-only", action="store_true", help="Load cached Hugging Face files only and avoid network access.")
    parser.add_argument("--epsilon", type=float, required=True, help="Target privacy epsilon for the analytic Gaussian calibration.")
    parser.add_argument("--delta-priv", type=float, required=True, help="Target privacy delta for the analytic Gaussian calibration.")
    parser.add_argument("--delta-mask", type=float, default=1e-3, help="Positive diagonal offset added before trace normalization.")
    parser.add_argument("--noise-scale-multiplier", type=float, default=1.0, help="Additional multiplier applied after left-scale construction for empirical utility sweeps.")
    parser.add_argument("--clip-norm", type=float, default=32.0, help="Frobenius norm clipping threshold for pre-noise vision tokens.")
    parser.add_argument(
        "--patch-alpha",
        type=float,
        default=0.70,
        help="Blend factor for patch pooling: alpha * max + (1 - alpha) * mean.",
    )
    parser.add_argument(
        "--privacy-map",
        type=Path,
        default=None,
        help="Optional path to a precomputed privacy_map.npy. If omitted, the script recomputes it from the image.",
    )

    parser.add_argument("--person-model", default="yolo11n.pt", help="YOLO model used for person detection.")
    parser.add_argument("--person-conf", type=float, default=0.25, help="Person detection confidence threshold.")
    parser.add_argument("--face-model", default=None, help="Optional auxiliary YOLO face/head detector.")
    parser.add_argument("--face-model-kind", choices=("face", "head"), default="face")
    parser.add_argument("--face-conf", type=float, default=0.20, help="Confidence threshold for the auxiliary face/head detector.")
    parser.add_argument(
        "--yunet-model",
        type=Path,
        default=Path("models/face_detection_yunet_2023mar.onnx"),
        help="YuNet model used before OpenCV cascade fallback.",
    )
    parser.add_argument("--yunet-score-threshold", type=float, default=0.55)
    parser.add_argument("--upper-body-weight", type=float, default=0.64)
    return parser.parse_args()


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _b_plus_function(v: float, epsilon: float) -> float:
    return normal_cdf(math.sqrt(epsilon * v)) - math.exp(epsilon) * normal_cdf(-math.sqrt(epsilon * (v + 2.0)))


def _b_minus_function(u: float, epsilon: float) -> float:
    return normal_cdf(-math.sqrt(epsilon * u)) - math.exp(epsilon) * normal_cdf(-math.sqrt(epsilon * (u + 2.0)))


def calibrate_analytic_matrix_gaussian(
    epsilon: float,
    delta_priv: float,
    sensitivity: float,
    iterations: int = 5000,
    upper_bound: float = 1e5,
) -> AnalyticGaussianCalibration:
    # This follows the public DP-Forward reference implementation for the analytic matrix Gaussian scalar.
    delta_threshold = normal_cdf(0.0) - math.exp(epsilon) * normal_cdf(-math.sqrt(2.0 * epsilon))
    start = 0.0
    end = upper_bound
    bound_fn = _b_plus_function if delta_priv >= delta_threshold else _b_minus_function

    for _ in range(iterations):
        midpoint = (start + end) / 2.0
        value = bound_fn(midpoint, epsilon)
        if value < delta_priv:
            end = midpoint
        else:
            start = midpoint

    solution = end
    if delta_priv >= delta_threshold:
        alpha = math.sqrt(1.0 + solution / 2.0) - math.sqrt(solution / 2.0)
    else:
        alpha = math.sqrt(1.0 + solution / 2.0) + math.sqrt(solution / 2.0)

    upper_bound_scalar = math.sqrt(2.0 * epsilon) / alpha
    base_noise_std = sensitivity / upper_bound_scalar
    return AnalyticGaussianCalibration(
        epsilon=epsilon,
        delta_priv=delta_priv,
        sensitivity=sensitivity,
        upper_bound_scalar=upper_bound_scalar,
        base_noise_std=base_noise_std,
        delta_threshold=delta_threshold,
    )


def clip_matrix_frobenius(hidden_states: torch.Tensor, clip_norm: float) -> tuple[torch.Tensor, ClipStats]:
    original_norm = float(torch.linalg.norm(hidden_states, ord="fro").item())
    if original_norm <= clip_norm or original_norm == 0.0:
        clipped = hidden_states
        clip_scale = 1.0
    else:
        clip_scale = clip_norm / original_norm
        clipped = hidden_states * clip_scale
    clipped_norm = float(torch.linalg.norm(clipped, ord="fro").item())
    return clipped, ClipStats(
        clip_norm=clip_norm,
        original_fro_norm=original_norm,
        clipped_fro_norm=clipped_norm,
        clip_scale=clip_scale,
    )


def build_patch_covariance_diagonal(mask_scores: torch.Tensor, delta_mask: float) -> torch.Tensor:
    weights = mask_scores.float().clamp(0.0, 1.0) + float(delta_mask)
    return weights * (weights.numel() / weights.sum())


def build_patch_noise_factors(
    patch_scores: torch.Tensor,
    epsilon: float,
    delta_priv: float,
    delta_mask: float,
    clip_norm: float,
    noise_scale_multiplier: float = 1.0,
) -> tuple[PatchNoiseFactors, AnalyticGaussianCalibration]:
    sensitivity = 2.0 * clip_norm
    calibration = calibrate_analytic_matrix_gaussian(
        epsilon=epsilon,
        delta_priv=delta_priv,
        sensitivity=sensitivity,
    )
    covariance_diag = build_patch_covariance_diagonal(patch_scores, delta_mask=delta_mask)
    base_left_factor_scale = calibration.base_noise_std
    left_factor_scale = base_left_factor_scale * float(noise_scale_multiplier)
    row_noise_scales = left_factor_scale * covariance_diag.sqrt()
    return (
        PatchNoiseFactors(
            covariance_diag=covariance_diag,
            row_noise_scales=row_noise_scales,
            required_min_singular_value=calibration.base_noise_std,
            left_factor_scale=left_factor_scale,
        ),
        calibration,
    )


def build_aux_detector(args: argparse.Namespace) -> tuple[object | None, bool, bool]:
    detectors: list[object] = []
    uses_opencv = False
    uses_yunet = False

    if args.face_model:
        detectors.append(prior.YoloAuxDetector(args.face_model, conf=args.face_conf, device=args.device, kind=args.face_model_kind))

    yunet_model_path = args.yunet_model
    if not yunet_model_path.is_absolute():
        yunet_model_path = PROJECT_ROOT / yunet_model_path
    yunet_detector = prior.YuNetFaceDetector(yunet_model_path, score_threshold=args.yunet_score_threshold)
    if yunet_detector.available:
        detectors.append(yunet_detector)
        uses_yunet = True

    opencv_detector = prior.OpenCvFaceDetector()
    if opencv_detector.available:
        detectors.append(opencv_detector)
        uses_opencv = True

    if not detectors:
        return None, uses_yunet, uses_opencv
    return prior.FallbackFaceDetector(detectors), uses_yunet, uses_opencv


def compute_privacy_map(args: argparse.Namespace, image_bgr: np.ndarray) -> np.ndarray:
    if args.privacy_map is not None:
        return np.load(args.privacy_map).astype(np.float32)

    person_model = YOLO(args.person_model)
    aux_detector, _, _ = build_aux_detector(args)
    person_boxes = prior.detect_person_boxes(person_model, image_bgr, conf=args.person_conf, device=args.device)
    records = [prior.build_detection_record(box, image_bgr, aux_detector, args.face_model_kind) for box in person_boxes]
    return prior.build_privacy_map(image_bgr.shape[:2], records, upper_body_weight=args.upper_body_weight)


def aggregate_scores_for_qwen_grid(
    privacy_map: np.ndarray,
    grid_thw: torch.Tensor,
    vision_patch_size: int,
    patch_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    grid_t, grid_h, grid_w = [int(v) for v in grid_thw[0].tolist()]
    if grid_t != 1:
        raise NotImplementedError("This prototype currently supports single-image inputs only.")

    resized_width = grid_w * vision_patch_size
    resized_height = grid_h * vision_patch_size
    resized_map = cv2.resize(privacy_map, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    patch_scores, _ = prior.aggregate_patch_scores(
        resized_map,
        patch_size=vision_patch_size,
        alpha=patch_alpha,
        sigma_min=0.0,
        sigma_max=1.0,
    )
    if patch_scores.shape != (grid_h, grid_w):
        raise ValueError(f"Unexpected patch score shape {patch_scores.shape}, expected {(grid_h, grid_w)}")
    return resized_map, patch_scores


def apply_patchwise_dp_noise(
    hidden_states: torch.Tensor,
    row_noise_scales: torch.Tensor,
    clip_norm: float,
) -> tuple[torch.Tensor, ClipStats]:
    clipped_hidden_states, clip_stats = clip_matrix_frobenius(hidden_states, clip_norm=clip_norm)
    noise = torch.randn_like(clipped_hidden_states) * row_noise_scales.to(
        device=clipped_hidden_states.device,
        dtype=clipped_hidden_states.dtype,
    ).unsqueeze(-1)
    return clipped_hidden_states + noise, clip_stats


class PatchwiseDPVisionController:
    def __init__(self, row_noise_scales: torch.Tensor, clip_norm: float) -> None:
        self.row_noise_scales = row_noise_scales.detach().clone().float()
        self.clip_norm = float(clip_norm)
        self.last_clip_stats: ClipStats | None = None

    def _build_outputs(
        self,
        visual_model: torch.nn.Module,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs,
    ) -> BaseModelOutputWithDeepstackFeatures:
        hidden_states = visual_model.patch_embed(hidden_states)
        if hidden_states.shape[0] != self.row_noise_scales.numel():
            raise ValueError(
                f"Row-scale length mismatch: got {self.row_noise_scales.numel()} scales for {hidden_states.shape[0]} visual tokens."
            )

        hidden_states, clip_stats = apply_patchwise_dp_noise(
            hidden_states=hidden_states,
            row_noise_scales=self.row_noise_scales,
            clip_norm=self.clip_norm,
        )
        self.last_clip_stats = clip_stats

        # Keep the absolute position signal deterministic and only perturb the
        # content-bearing patch embeddings.
        pos_embeds = visual_model.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = visual_model.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists: list[torch.Tensor] = []
        for layer_num, block in enumerate(visual_model.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in visual_model.deepstack_visual_indexes:
                idx = visual_model.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = visual_model.deepstack_merger_list[idx](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        merged_hidden_states = visual_model.merger(hidden_states)
        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
            deepstack_features=deepstack_feature_lists,
        )

    def forward(
        self,
        visual_model: torch.nn.Module,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs,
    ) -> BaseModelOutputWithDeepstackFeatures | tuple:
        return_dict = kwargs.pop("return_dict", True)
        outputs = self._build_outputs(
            visual_model=visual_model,
            hidden_states=hidden_states,
            grid_thw=grid_thw,
            **kwargs,
        )
        return outputs if return_dict else outputs.to_tuple()

    @contextmanager
    def install_on(self, visual_model: torch.nn.Module):
        original_forward = visual_model.forward
        controller = self

        def patched_forward(module_self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
            return controller.forward(
                visual_model=module_self,
                hidden_states=hidden_states,
                grid_thw=grid_thw,
                **kwargs,
            )

        visual_model.forward = MethodType(patched_forward, visual_model)
        try:
            yield visual_model
        finally:
            visual_model.forward = original_forward

    def require_clip_stats(self) -> ClipStats:
        if self.last_clip_stats is None:
            raise RuntimeError("Clip stats are not available before running the DP vision forward path.")
        return self.last_clip_stats


def save_patch_overlay(image_bgr: np.ndarray, patch_scores: np.ndarray, destination: Path) -> None:
    upsampled = cv2.resize(patch_scores.astype(np.float32), (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    heatmap = cv2.applyColorMap(np.clip(upsampled * 255.0, 0.0, 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(image_bgr, 0.60, heatmap, 0.40, 0.0)
    cv2.imwrite(str(destination), overlay)


def build_artifacts(
    calibration: AnalyticGaussianCalibration,
    clip_stats: ClipStats,
    patch_scores: torch.Tensor,
    covariance_diag: torch.Tensor,
    row_noise_scales: torch.Tensor,
    required_min_singular_value: float,
    left_factor_scale: float,
    grid_thw: torch.Tensor,
    visual_hidden_size: int,
    model_id: str,
    delta_mask: float,
    noise_scale_multiplier: float,
    mode: str,
) -> DPForwardArtifacts:
    grid_t, grid_h, grid_w = [int(v) for v in grid_thw[0].tolist()]
    return DPForwardArtifacts(
        calibration=calibration,
        clip_stats=clip_stats,
        patch_noise=PatchNoiseSummary(
            num_patches=int(patch_scores.numel()),
            grid_t=grid_t,
            grid_h=grid_h,
            grid_w=grid_w,
            delta_mask=delta_mask,
            noise_scale_multiplier=noise_scale_multiplier,
            required_min_singular_value=required_min_singular_value,
            left_factor_scale=left_factor_scale,
            trace_normalized_sum=float(covariance_diag.sum().item()),
            min_patch_score=float(patch_scores.min().item()),
            max_patch_score=float(patch_scores.max().item()),
            min_cov_diag=float(covariance_diag.min().item()),
            max_cov_diag=float(covariance_diag.max().item()),
            min_row_scale=float(row_noise_scales.min().item()),
            max_row_scale=float(row_noise_scales.max().item()),
        ),
        visual_hidden_size=visual_hidden_size,
        model_id=model_id,
        mode=mode,
    )


def dtype_from_name(name: str, mode: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name == "auto":
        return torch.float32 if mode == "vision-only" else torch.float32
    return mapping[name]


def pretrained_dtype_arg(name: str):
    if name == "auto":
        return "auto"
    return dtype_from_name(name, mode="full-model")


def resolve_pretrained_source(model_id: str, local_files_only: bool) -> str:
    if not local_files_only:
        return model_id

    candidate = Path(model_id)
    if candidate.exists():
        return str(candidate)

    if "/" not in model_id:
        return model_id

    owner, repo = model_id.split("/", 1)
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{owner}--{repo}"
    ref_path = cache_root / "refs" / "main"
    if not ref_path.exists():
        return model_id

    revision = ref_path.read_text(encoding="utf-8").strip()
    snapshot_dir = cache_root / "snapshots" / revision
    return str(snapshot_dir) if snapshot_dir.exists() else model_id


def load_visual_state_from_checkpoint(model_root: Path) -> dict[str, torch.Tensor]:
    index_path = model_root / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing sharded index: {index_path}")

    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index_data["weight_map"]
    visual_items = [(key, shard) for key, shard in weight_map.items() if key.startswith("model.visual.")]
    if not visual_items:
        raise RuntimeError(f"No visual tower weights were found in {index_path}")

    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in visual_items:
        shard_to_keys.setdefault(shard, []).append(key)

    state_dict: dict[str, torch.Tensor] = {}
    for shard, full_keys in shard_to_keys.items():
        with safe_open(model_root / shard, framework="pt", device="cpu") as handle:
            for full_key in full_keys:
                state_dict[full_key.removeprefix("model.visual.")] = handle.get_tensor(full_key)
    return state_dict


def load_visual_model_from_checkpoint(
    model_root: Path,
    config,
    device: torch.device,
    dtype: torch.dtype,
) -> Qwen3VLVisionModel:
    visual_model = Qwen3VLVisionModel(config.vision_config)
    visual_state = load_visual_state_from_checkpoint(model_root)
    missing_keys, unexpected_keys = visual_model.load_state_dict(visual_state, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Unexpected visual checkpoint mismatch. Missing={missing_keys[:5]}, unexpected={unexpected_keys[:5]}"
        )
    visual_model = visual_model.to(device=device, dtype=dtype)
    visual_model.eval()
    return visual_model


def save_shape_array(destination: Path, value) -> None:
    if isinstance(value, torch.Size):
        array = np.array(list(value), dtype=np.int32)
    elif isinstance(value, torch.Tensor):
        array = np.array(list(value.shape), dtype=np.int32)
    elif isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        array = np.array([list(t.shape) for t in value], dtype=np.int32)
    else:
        array = np.array(value, dtype=np.int32)
    np.save(destination, array)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    image_pil = Image.open(args.image).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    source_id = resolve_pretrained_source(args.model_id, args.local_files_only)
    hf_kwargs = {"local_files_only": args.local_files_only}
    image_processor = AutoImageProcessor.from_pretrained(source_id, **hf_kwargs)
    image_inputs = image_processor(images=image_pil, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"]
    image_grid_thw = image_inputs["image_grid_thw"]

    config = AutoConfig.from_pretrained(source_id, **hf_kwargs)
    vision_patch_size = int(config.vision_config.patch_size)
    privacy_map = compute_privacy_map(args, image_bgr)
    resized_map, patch_scores_np = aggregate_scores_for_qwen_grid(
        privacy_map=privacy_map,
        grid_thw=image_grid_thw,
        vision_patch_size=vision_patch_size,
        patch_alpha=args.patch_alpha,
    )
    patch_scores = torch.from_numpy(patch_scores_np.reshape(-1))
    patch_noise, calibration = build_patch_noise_factors(
        patch_scores=patch_scores,
        epsilon=args.epsilon,
        delta_priv=args.delta_priv,
        delta_mask=args.delta_mask,
        clip_norm=args.clip_norm,
        noise_scale_multiplier=args.noise_scale_multiplier,
    )

    dtype = dtype_from_name(args.dtype, mode=args.mode)
    device = torch.device(args.device)
    output_dir = args.output / args.image.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        seq_len = patch_scores.numel()
        hidden_states = torch.randn(seq_len, int(config.vision_config.hidden_size), dtype=dtype, device=device)
        _, clip_stats = apply_patchwise_dp_noise(
            hidden_states=hidden_states,
            row_noise_scales=patch_noise.row_noise_scales.to(device=device, dtype=dtype),
            clip_norm=args.clip_norm,
        )
        np.save(output_dir / "dry_run_hidden_shape.npy", np.array(hidden_states.shape, dtype=np.int32))
    else:
        controller = PatchwiseDPVisionController(
            row_noise_scales=patch_noise.row_noise_scales,
            clip_norm=args.clip_norm,
        )
        image_grid_thw = image_grid_thw.to(device=device)

        if args.mode == "vision-only":
            model_root = Path(source_id)
            if not model_root.exists():
                raise FileNotFoundError(
                    f"Vision-only mode requires a local checkpoint directory, but {model_root} does not exist."
                )
            visual_model = load_visual_model_from_checkpoint(
                model_root=model_root,
                config=config,
                device=device,
                dtype=dtype,
            )
            pixel_values = pixel_values.to(device=device, dtype=visual_model.dtype)
            with torch.inference_mode():
                vision_output = controller.forward(
                    visual_model=visual_model,
                    hidden_states=pixel_values,
                    grid_thw=image_grid_thw,
                    return_dict=True,
                )
            clip_stats = controller.require_clip_stats()
        else:
            try:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    source_id,
                    torch_dtype=pretrained_dtype_arg(args.dtype),
                    low_cpu_mem_usage=True,
                    **hf_kwargs,
                ).to(device)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load the full Qwen3-VL model. On CPU-only machines, try --mode vision-only first."
                ) from exc

            model.eval()
            pixel_values = pixel_values.to(device=device, dtype=model.visual.dtype)
            with controller.install_on(model.visual):
                with torch.inference_mode():
                    vision_output = model.get_image_features(
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        return_dict=True,
                    )
            clip_stats = controller.require_clip_stats()

        save_shape_array(output_dir / "vision_pooler_shape.npy", vision_output.pooler_output)
        save_shape_array(output_dir / "vision_last_hidden_shape.npy", vision_output.last_hidden_state)

    artifacts = build_artifacts(
        calibration=calibration,
        clip_stats=clip_stats,
        patch_scores=patch_scores,
        covariance_diag=patch_noise.covariance_diag,
        row_noise_scales=patch_noise.row_noise_scales,
        required_min_singular_value=patch_noise.required_min_singular_value,
        left_factor_scale=patch_noise.left_factor_scale,
        grid_thw=image_grid_thw,
        visual_hidden_size=int(config.vision_config.hidden_size),
        model_id=args.model_id,
        delta_mask=args.delta_mask,
        noise_scale_multiplier=args.noise_scale_multiplier,
        mode=args.mode,
    )

    np.save(output_dir / "privacy_map.npy", privacy_map.astype(np.float32))
    np.save(output_dir / "privacy_map_resized_for_qwen.npy", resized_map.astype(np.float32))
    np.save(output_dir / "qwen_patch_scores.npy", patch_scores_np.astype(np.float32))
    np.save(output_dir / "qwen_patch_cov_diag.npy", patch_noise.covariance_diag.cpu().numpy().astype(np.float32))
    np.save(output_dir / "qwen_row_noise_scales.npy", patch_noise.row_noise_scales.cpu().numpy().astype(np.float32))
    save_patch_overlay(image_bgr, patch_scores_np, output_dir / "qwen_patch_scores_overlay.jpg")
    (output_dir / "dp_metadata.json").write_text(json.dumps(asdict(artifacts), indent=2), encoding="utf-8")

    print(
        f"{args.image}: mode={args.mode}, grid {tuple(image_grid_thw[0].tolist())}, "
        f"patches {patch_scores.numel()}, "
        f"base_noise_std={artifacts.calibration.base_noise_std:.6f}, "
        f"noise_scale_multiplier={artifacts.patch_noise.noise_scale_multiplier:.6f}, "
        f"left_factor_scale={artifacts.patch_noise.left_factor_scale:.6f}, "
        f"row_scale_range=({artifacts.patch_noise.min_row_scale:.6f}, {artifacts.patch_noise.max_row_scale:.6f}), "
        f"saved to {output_dir}"
    )


if __name__ == "__main__":
    main()


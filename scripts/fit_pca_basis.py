"""Offline PCA-basis fit tool for the DP vision-tower hook.

Runs the VLM on a batch of images, captures feature tensors at the chosen
hook location(s), and fits a PCA basis per-feature-dim (Welford/Chan streaming
stats → torch.linalg.eigh → top-rank eigenvectors). Writes per-file `.pt`s
using the exact schema `dp/fixed_pca.py:_load_fixed_pca_basis` consumes:

    single-dim location:  <out>/<safe_loc>.pt
    multi-dim location:   <out>/<safe_loc>__dim<N>.pt

Each file: {mean[1,dim], basis[rank,dim], rank, effective_rank, dim,
           explained_variance_ratio}.

Privacy note: the basis is fit from the same training data the noise later
"protects." This pipeline is local-DP noise with a data-dependent utility knob.
Formal central-DP over the basis itself is out of scope.

Usage:
  python scripts/fit_pca_basis.py \
      --model /path/to/Qwen3-VL-8B-Instruct \
      --locations vit_hidden_8,vit_output \
      --rank 128 --samples 256 \
      --images-dir /path/to/jpgs \
      --out pca_basis/run0/

If --images-dir is omitted, falls back to --num-synthetic random images (for
smoke-testing the fit pipeline; not for real training).
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _safe_loc_name(location: str) -> str:
    # Inlined from dp/_common.py to avoid triggering dp.apply_patch() as a side
    # effect of importing dp — the fit tool must run against a clean, un-hooked
    # model.
    return location.replace("/", "__")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model directory for Qwen3-VL")
    ap.add_argument(
        "--locations",
        required=True,
        help="Comma-separated noise locations (e.g. vit_hidden_8 or vit_hidden_16,vit_output)",
    )
    ap.add_argument("--rank", type=int, default=128, help="Target PCA basis rank")
    ap.add_argument("--samples", type=int, default=256, help="Number of images to pass through")
    ap.add_argument("--images-dir", default=None,
                    help="Directory of JPG/PNG images. If omitted, synthesizes random images.")
    ap.add_argument(
        "--train-files",
        default=None,
        help="Comma-separated parquet files for default sampling when --images-dir is not set.",
    )
    ap.add_argument("--seed", type=int, default=1, help="Random seed for reproducible sampling.")
    ap.add_argument("--num-synthetic", type=int, default=32,
                    help="Number of synthetic images when --images-dir is omitted (capped at --samples).")
    ap.add_argument("--image-size", type=int, default=336)
    ap.add_argument("--out", required=True, help="Output dir for <safe_loc>[.__dim<N>].pt files")
    ap.add_argument("--prompt", default="Describe this image.")
    ap.add_argument("--max-tokens-per-batch", type=int, default=0,
                    help="Subsample at most N tokens into the stats per batch "
                         "(0 = use all tokens). Bounds M2 update cost for 4096-dim.")
    return ap.parse_args()


# =============================================================================
# Welford/Chan streaming stats per (location, feat_dim)
# =============================================================================
class StreamingStats:
    """Single-dim streaming mean + scatter matrix (Welford/Chan parallel combine).

    All accumulation in fp32 on CPU (avoid GPU OOM when dim=4096 and we keep
    [4096,4096] live during a long fit). The number of tokens goes up to ~10^7
    — Chan's parallel combine keeps this stable.
    """

    def __init__(self, dim: int):
        self.dim = int(dim)
        self.count = 0
        self.mean = torch.zeros(1, dim, dtype=torch.float32)
        self.M2 = torch.zeros(dim, dim, dtype=torch.float32)

    def update(self, batch: torch.Tensor):
        """batch: [N, dim] fp32 on CPU."""
        if batch.numel() == 0:
            return
        n_b = batch.shape[0]
        if n_b == 0:
            return
        mean_b = batch.mean(dim=0, keepdim=True)
        centered = batch - mean_b
        M2_b = centered.t() @ centered  # [dim, dim]

        if self.count == 0:
            self.count = n_b
            self.mean.copy_(mean_b)
            self.M2.copy_(M2_b)
            return

        n_a = self.count
        n = n_a + n_b
        delta = mean_b - self.mean  # [1, dim]
        new_mean = self.mean + delta * (n_b / n)
        # parallel combine for M2 (Chan/Golub/LeVeque): M2 = M2_a + M2_b + delta^T delta * (n_a*n_b/n)
        self.M2.add_(M2_b)
        self.M2.add_((delta.t() @ delta) * (n_a * n_b / n))
        self.mean.copy_(new_mean)
        self.count = n

    def fit_basis(self, rank: int, device: str = "cuda"):
        if self.count < 2:
            raise RuntimeError(
                f"streaming stats has count={self.count} < 2; not enough tokens to fit PCA"
            )
        if rank > self.dim:
            raise RuntimeError(f"rank={rank} > feat dim={self.dim}")

        cov = (self.M2 / (self.count - 1)).to(device=device, dtype=torch.float32)
        # symmetrize defensively
        cov = 0.5 * (cov + cov.t())
        eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending eigenvalues
        eigvals = eigvals.flip(0)
        eigvecs = eigvecs.flip(1)  # columns = eigenvectors

        top_vecs = eigvecs[:, :rank].t().contiguous()  # [rank, dim]
        top_vals = eigvals[:rank]

        total_var = float(torch.clamp(eigvals.sum(), min=1e-12).item())
        evr = float((top_vals.sum() / total_var).item())

        return {
            "mean": self.mean.clone().cpu(),
            "basis": top_vecs.cpu(),
            "rank": int(rank),
            "effective_rank": int(rank),
            "dim": int(self.dim),
            "explained_variance_ratio": evr,
        }


# =============================================================================
# Image loading
# =============================================================================
def _load_images(
    images_dir: str | None, num_synthetic: int, image_size: int, samples: int, seed: int
):
    images = []
    if images_dir:
        p = Path(images_dir)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = sorted(x for x in p.rglob("*") if x.suffix.lower() in exts)
        files = files[:samples]
        if not files:
            raise RuntimeError(f"no images found under {images_dir}")
        for f in files:
            try:
                img = Image.open(f).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"[fit] skip {f}: {e}")
        print(f"[fit] loaded {len(images)} images from {images_dir}")
    else:
        n = min(num_synthetic, samples)
        rng = np.random.default_rng(seed)
        for i in range(n):
            arr = rng.integers(0, 256, size=(image_size, image_size, 3), dtype="uint8")
            images.append(Image.fromarray(arr))
        print(f"[fit] SYNTHETIC mode: generated {n} random images "
              f"(for real fits, pass --images-dir)")
    return images


IMAGE_COL_CANDIDATES = [
    "images",
    "image",
    "image_path",
    "img_path",
    "img",
    "path",
    "file",
    "filepath",
]


def _first_existing_column(columns, preferred):
    mapping = {str(c).lower(): c for c in columns}
    for col in preferred:
        hit = mapping.get(col.lower())
        if hit is not None:
            return hit
    return None


def _decode_image_value(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, str) and value.strip():
        return Image.open(value).convert("RGB")
    if isinstance(value, np.ndarray):
        if value.dtype == object and value.size > 0:
            return _decode_image_value(value.flat[0])
        # bytes-like parquet payload may appear as ndarray(uint8)
        if value.ndim == 1:
            return Image.open(io.BytesIO(value.tobytes())).convert("RGB")
        return Image.fromarray(value).convert("RGB")
    if isinstance(value, dict):
        if "bytes" in value and value["bytes"] is not None:
            raw = value["bytes"]
            if isinstance(raw, np.ndarray):
                raw = raw.tobytes()
            if isinstance(raw, (bytes, bytearray)):
                return Image.open(io.BytesIO(raw)).convert("RGB")
        if "path" in value and value["path"]:
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict):
            return _decode_image_value(first)
    raise ValueError(f"Unsupported image value type: {type(value)}")


def _round_robin_counts(total: int, n_files: int):
    base = total // n_files
    rem = total % n_files
    counts = [base] * n_files
    for i in range(rem):
        counts[i] += 1
    return counts


def _load_images_from_train_parquets(train_files: str, samples: int, seed: int):
    files = [s.strip() for s in str(train_files).split(",") if s.strip()]
    if not files:
        raise RuntimeError("--train-files is empty")

    counts = _round_robin_counts(samples, len(files))
    print(f"[fit] parquet sampling: files={len(files)} samples={samples} seed={seed}")
    for i, f in enumerate(files):
        print(f"[fit] target allocation: file[{i}] -> {counts[i]} samples | {f}")

    states = []
    for i, f in enumerate(files):
        df = pd.read_parquet(f)
        img_col = _first_existing_column(df.columns, IMAGE_COL_CANDIDATES)
        if img_col is None:
            raise RuntimeError(f"no image column found in {f}; columns={list(df.columns)}")
        rng = np.random.default_rng(seed + i * 10007)
        perm = rng.permutation(len(df))
        states.append(
            {
                "file": f,
                "df": df,
                "img_col": img_col,
                "perm": perm,
                "cursor": 0,
                "taken": 0,
                "failed_decode": 0,
            }
        )

    def take_from_state(state, need):
        out = []
        while len(out) < need and state["cursor"] < len(state["perm"]):
            row_idx = int(state["perm"][state["cursor"]])
            state["cursor"] += 1
            try:
                img = _decode_image_value(state["df"].iloc[row_idx][state["img_col"]])
                out.append(img)
                state["taken"] += 1
            except Exception:
                state["failed_decode"] += 1
        return out

    images = []
    for i, state in enumerate(states):
        images.extend(take_from_state(state, counts[i]))

    deficit = samples - len(images)
    # Refill deficits in round-robin order from remaining rows.
    while deficit > 0:
        progress = False
        for state in states:
            if deficit <= 0:
                break
            more = take_from_state(state, 1)
            if more:
                images.extend(more)
                deficit -= 1
                progress = True
        if not progress:
            break

    for state in states:
        print(
            f"[fit] sampled {state['taken']} images from {state['file']} "
            f"(decode_failed={state['failed_decode']}, rows={len(state['df'])})"
        )

    if len(images) < samples:
        print(
            f"[fit] WARNING: requested samples={samples}, actual decoded images={len(images)}; "
            "continuing with fewer images."
        )
    return images


# =============================================================================
# Capture hooks
# =============================================================================
def _install_capture_hooks(model, locations, sinks):
    """Register forward hooks that push each encountered tensor into the sink.

    sinks: {location: [tensor, ...]} mutated in place. Tensors are moved to CPU
    fp32 so the GPU memory footprint stays flat across many batches.

    Location semantics mirror dp/patch.py:_install_hook_if_needed.
    """
    # locate vision tower
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        vision_tower = model.model.visual
    elif hasattr(model, "visual"):
        vision_tower = model.visual
    elif hasattr(model, "vision_tower"):
        vision_tower = model.vision_tower
    else:
        raise RuntimeError("cannot locate vision tower on model")

    n_layers = None
    if hasattr(vision_tower, "blocks"):
        n_layers = len(vision_tower.blocks)
    elif hasattr(vision_tower, "encoder") and hasattr(vision_tower.encoder, "layers"):
        n_layers = len(vision_tower.encoder.layers)

    handles = []

    def _make_sink_fn(loc):
        def _sink(t):
            if torch.is_tensor(t) and t.is_floating_point() and t.ndim >= 2:
                sinks[loc].append(t.detach().cpu().float())
        return _sink

    def _tree_push(loc, obj):
        sink_fn = _make_sink_fn(loc)
        if torch.is_tensor(obj):
            sink_fn(obj)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _tree_push(loc, v)
        else:
            # ModelOutput-like
            for attr in ("last_hidden_state", "pooler_output", "hidden_states",
                         "deepstack_features"):
                v = getattr(obj, attr, None)
                if v is not None:
                    _tree_push(loc, v)

    for loc in locations:
        sinks.setdefault(loc, [])
        if loc == "vit_input":
            def hook(_m, inp, loc=loc):
                for x in inp:
                    _tree_push(loc, x)
            handles.append(vision_tower.patch_embed.register_forward_pre_hook(hook))
        elif loc == "vit_after_pos_embed":
            def hook(_m, inp, loc=loc):
                for x in inp:
                    _tree_push(loc, x)
            handles.append(vision_tower.blocks[0].register_forward_pre_hook(hook))
        elif loc == "vit_output":
            def hook(_m, _inp, output, loc=loc):
                _tree_push(loc, output)
            handles.append(vision_tower.register_forward_hook(hook))
        elif loc.startswith("vit_hidden_"):
            idx = int(loc.split("_")[-1])
            if n_layers is None or idx >= n_layers:
                raise RuntimeError(f"invalid {loc}: have {n_layers} layers")

            def hook_block(_m, _inp, output, loc=loc):
                _tree_push(loc, output)
            handles.append(vision_tower.blocks[idx].register_forward_hook(hook_block))

            # Mirror the deepstack-backfill hook from dp/patch.py — so we also
            # fit bases for the post-merger 4096-dim features that patch.py will
            # noise at inference/train time.
            ds_idx = getattr(vision_tower, "deepstack_visual_indexes", None)
            if ds_idx is None and hasattr(vision_tower, "config"):
                ds_idx = getattr(vision_tower.config, "deepstack_visual_indexes", None)
            earlier = [i for i in (ds_idx or []) if i < idx]
            if earlier:
                def hook_backfill(_m, _inp, output, loc=loc, ds_idx=ds_idx):
                    feats = getattr(output, "deepstack_features", None)
                    if feats is None:
                        return
                    sink_fn = _make_sink_fn(loc)
                    for list_pos, capture_idx in enumerate(ds_idx):
                        if capture_idx in earlier and list_pos < len(feats):
                            sink_fn(feats[list_pos])
                handles.append(vision_tower.register_forward_hook(hook_backfill))
        else:
            raise RuntimeError(f"unknown location: {loc}")

    return handles


# =============================================================================
# Drive the forward passes
# =============================================================================
def _collect_features(
    model, processor, images, prompt, locations, max_tokens_per_batch, seed: int
):
    device = next(model.parameters()).device
    token_sample_gen = torch.Generator(device="cpu")
    token_sample_gen.manual_seed(int(seed))

    # per-location per-dim streaming stats
    stats: dict[str, dict[int, StreamingStats]] = {loc: {} for loc in locations}
    sinks: dict[str, list[torch.Tensor]] = {loc: [] for loc in locations}
    handles = _install_capture_hooks(model, locations, sinks)

    try:
        for i, img in enumerate(images):
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]}]
            chat_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=[chat_text], images=[img], return_tensors="pt")
            inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

            with torch.no_grad():
                # forward through the visual tower only — don't need the LM side
                pixel_values = inputs.get("pixel_values")
                grid_thw = inputs.get("image_grid_thw")
                if pixel_values is None:
                    raise RuntimeError("processor did not return pixel_values")
                # call model.model.visual or model.visual directly to minimize cost
                vt = (model.model.visual if hasattr(model, "model") and hasattr(model.model, "visual")
                      else model.visual)
                try:
                    _ = vt(pixel_values, grid_thw=grid_thw) if grid_thw is not None else vt(pixel_values)
                except TypeError:
                    _ = vt(pixel_values)

            # drain sinks into streaming stats
            for loc in locations:
                bucket = sinks[loc]
                sinks[loc] = []
                for t in bucket:
                    flat = t.reshape(-1, t.shape[-1])  # [N, dim]
                    dim = flat.shape[1]
                    if max_tokens_per_batch > 0 and flat.shape[0] > max_tokens_per_batch:
                        idx = torch.randperm(
                            flat.shape[0], generator=token_sample_gen
                        )[:max_tokens_per_batch]
                        flat = flat[idx]
                    st = stats[loc].setdefault(dim, StreamingStats(dim))
                    st.update(flat)

            if (i + 1) % max(1, len(images) // 10) == 0 or i == len(images) - 1:
                msg = [f"[fit] {i+1}/{len(images)} images"]
                for loc in locations:
                    dims = sorted(stats[loc].keys())
                    counts = {d: stats[loc][d].count for d in dims}
                    msg.append(f"{loc}={counts}")
                print(" | ".join(msg))
    finally:
        for h in handles:
            h.remove()

    return stats


# =============================================================================
# Save per-file
# =============================================================================
def _save_bases(stats, rank: int, out_dir: Path, device: str = "cuda"):
    out_dir.mkdir(parents=True, exist_ok=True)
    for loc, dim_map in stats.items():
        safe = _safe_loc_name(loc)
        dims = sorted(dim_map.keys())

        if not dims:
            raise RuntimeError(f"no tokens collected for {loc}")

        if len(dims) == 1:
            dim = dims[0]
            entry = dim_map[dim].fit_basis(rank, device=device)
            path = out_dir / f"{safe}.pt"
            torch.save(entry, path)
            print(f"[fit] wrote {path} | dim={dim} | rank={entry['rank']} | "
                  f"evr={entry['explained_variance_ratio']:.4f} | "
                  f"tokens={dim_map[dim].count}")
        else:
            # multi-dim at the same location (e.g. vit_output)
            for dim in dims:
                entry = dim_map[dim].fit_basis(rank, device=device)
                path = out_dir / f"{safe}__dim{dim}.pt"
                torch.save(entry, path)
                print(f"[fit] wrote {path} | dim={dim} | rank={entry['rank']} | "
                      f"evr={entry['explained_variance_ratio']:.4f} | "
                      f"tokens={dim_map[dim].count}")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    locations = [s.strip() for s in args.locations.split(",") if s.strip()]
    print(f"[fit] model={args.model}")
    print(f"[fit] locations={locations}")
    print(f"[fit] rank={args.rank} samples={args.samples}")
    print(f"[fit] seed={args.seed}")

    # load model WITHOUT dp patch (we don't want noise during fit)
    # By importing transformers only, we bypass the dp.apply_patch monkey-patch.
    # (Running `python -c "import dp"` elsewhere would install the patch for THIS
    # process. We do not import dp above at module-import time.)
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    t0 = time.time()
    print("[fit] loading model (bf16, balanced across all visible GPUs)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[fit] model loaded in {time.time()-t0:.1f}s")

    # sanity: confirm dp patch is NOT installed on this model
    if getattr(model, "_vision_dp_everywhere_hook_installed", False):
        raise RuntimeError(
            "DP patch appears installed on the model — fit must run against a clean model."
        )

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    if args.images_dir:
        images = _load_images(
            args.images_dir, args.num_synthetic, args.image_size, args.samples, args.seed
        )
    elif args.train_files:
        images = _load_images_from_train_parquets(
            train_files=args.train_files, samples=args.samples, seed=args.seed
        )
    else:
        images = _load_images(
            args.images_dir, args.num_synthetic, args.image_size, args.samples, args.seed
        )

    stats = _collect_features(
        model=model,
        processor=processor,
        images=images,
        prompt=args.prompt,
        locations=locations,
        max_tokens_per_batch=args.max_tokens_per_batch,
        seed=args.seed,
    )

    # summary before fitting
    print("\n[fit] === stats summary ===")
    for loc, dim_map in stats.items():
        for dim, st in sorted(dim_map.items()):
            print(f"  {loc} dim={dim} tokens={st.count} "
                  f"mean_norm={st.mean.norm().item():.4f} "
                  f"M2_fro={st.M2.norm().item():.4e}")

    print(f"\n[fit] fitting bases (rank={args.rank})...")
    out_dir = Path(args.out)
    _save_bases(stats, rank=args.rank, out_dir=out_dir, device="cuda")
    print(f"\n[fit] done. bases written under {out_dir.resolve()}")


if __name__ == "__main__":
    main()

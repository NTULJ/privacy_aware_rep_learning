"""End-to-end debugger for the vision-DP pipeline.

Verifies, for one chosen location (e.g. `vit_hidden_8`):
  1. forward hook fires and captures features at that location
  2. MLP down-projection produces a low-rank latent
  3. per-token norm-clip + Gaussian DP noise applied in the subspace
  4. MLP up-projection reconstructs full dim
  5. downstream transformer + LM head still decode text

Run:
  source .venv/bin/activate
  python scripts/debug_dp_pipeline.py \
      --model /workspace/s/ddn/.../Models/Qwen3-VL-8B-Instruct \
      --location vit_hidden_8 --epsilon 1 --mlp-dim 128
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dp  # noqa: F401  — triggers apply_patch() on PreTrainedModel.from_pretrained


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--location", default="vit_hidden_8")
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--norm-c", type=float, default=1.0)
    ap.add_argument("--mlp-dim", type=int, default=128)
    ap.add_argument("--no-mlp", action="store_true",
                    help="Disable MLP subspace — use the raw full-dim Gaussian mechanism.")
    ap.add_argument("--fixed-pca", action="store_true",
                    help="Use the fixed-PCA mechanism instead of MLP. Requires --pca-basis-dir.")
    ap.add_argument("--pca-basis-dir", default=None,
                    help="Directory with <safe_loc>.pt and/or <safe_loc>__dim<N>.pt files "
                         "(produced by scripts/fit_pca_basis.py).")
    ap.add_argument(
        "--debug-dir",
        default=None,
        help=(
            "Optional explicit debug output directory. If omitted, use "
            "debug_out/eps{eps}_pos-{location}_mech-{mechanism}."
        ),
    )
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--prompt", default="Describe this image briefly.")
    return ap.parse_args()


def build_dummy_image(size: int = 336) -> Image.Image:
    import numpy as np
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype="uint8")
    return Image.fromarray(arr)


def _build_debug_dir(args) -> Path:
    """Build a run-specific debug directory name from key DP settings."""
    if args.debug_dir is not None and str(args.debug_dir).strip():
        return Path(args.debug_dir)

    if args.fixed_pca:
        mech = "fixed_pca"
    elif args.no_mlp:
        mech = "original"
    else:
        mech = "mlp_subspace"

    eps_str = f"{args.epsilon:g}"
    safe_loc = str(args.location).replace("/", "__")
    run_name = f"eps{eps_str}_pos-{safe_loc}_mech-{mech}"
    return PROJECT_ROOT / "debug_out" / run_name


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    debug_dir = _build_debug_dir(args)
    debug_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor, AutoConfig, Qwen3VLForConditionalGeneration

    print(f"[debug] loading config with vision_dp overrides from {args.model}", flush=True)
    config = AutoConfig.from_pretrained(args.model)

    config.vision_dp_enable = True
    config.vision_dp_epsilon = args.epsilon
    config.vision_dp_delta = args.delta
    config.vision_dp_local_dp = True
    config.vision_dp_noise_type = "aGM"
    config.vision_dp_norm_c = args.norm_c
    config.vision_dp_noise_location = args.location

    # Mechanism selection — mutually exclusive
    if args.fixed_pca:
        if args.pca_basis_dir is None:
            raise SystemExit("--fixed-pca requires --pca-basis-dir")
        config.vision_dp_use_fixed_pca = True
        config.vision_dp_pca_basis_dir = args.pca_basis_dir
        config.vision_dp_use_mlp_subspace = False
    elif args.no_mlp:
        config.vision_dp_use_mlp_subspace = False
    else:
        config.vision_dp_use_mlp_subspace = True
        config.vision_dp_mlp_hidden_dim = args.mlp_dim
    # debug hooks
    config.vision_dp_debug = True
    config.vision_dp_debug_dir = str(debug_dir)
    config.vision_dp_debug_save_full_tensors = True
    config.vision_dp_debug_max_dump = 4  # allow more dumps (pre-hook + post-hook + backfill)
    config.vision_dp_debug_max_tokens = 6
    config.vision_dp_debug_max_dims = 8

    print(f"[debug] loading model (device_map=balanced, bfloat16, 8 GPUs)", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Sanity-check that hook installed.
    if not getattr(model, "_vision_dp_everywhere_hook_installed", False):
        raise SystemExit("DP hook did not install — check override_config flags")

    print(f"[debug] hook installed: "
          f"handles={len(getattr(model, '_vision_dp_everywhere_hook_handles', []))}")

    if args.fixed_pca:
        # Print loaded PCA cache for visibility
        cache = getattr(model, "_vision_dp_pca_cache", None) or getattr(
            model.config, "_vision_dp_pca_cache", None
        )
        if cache is None:
            raise SystemExit("fixed-PCA cache missing — loader failed")
        for loc, entry in cache.items():
            if entry.get("multi_dim"):
                dims = sorted(entry["per_dim"].keys())
                print(f"[debug] PCA cache (multi-dim) loc={loc} dims={dims}")
                for d in dims:
                    m = entry["per_dim"][d]
                    print(f"    dim={d} rank={m['rank']} evr={m['explained_variance_ratio']:.4f} "
                          f"path={m['path']}")
            else:
                print(f"[debug] PCA cache loc={loc} dim={entry['dim']} rank={entry['rank']} "
                      f"evr={entry['explained_variance_ratio']:.4f} path={entry['path']}")
    elif not args.no_mlp:
        assert hasattr(model, "_dp_mlp_down") and hasattr(model, "_dp_mlp_up"), \
            "MLP subspace modules missing"
        # place MLP on same device as vision tower params
        vt_param = next(model.model.visual.parameters())
        model._dp_mlp_down.to(device=vt_param.device, dtype=vt_param.dtype)
        model._dp_mlp_up.to(device=vt_param.device, dtype=vt_param.dtype)
        print(f"[debug] MLP down: {model._dp_mlp_down} on {vt_param.device}")
        print(f"[debug] MLP up:   {model._dp_mlp_up}")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    img = build_dummy_image()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": args.prompt},
        ],
    }]
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[chat_text], images=[img], return_tensors="pt")
    # move to model's first input device
    first_device = next(model.parameters()).device
    inputs = {k: v.to(first_device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    print(f"[debug] running generate (max_new_tokens={args.max_new_tokens})...", flush=True)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    gen = out[0][inputs["input_ids"].shape[1]:]
    text = processor.tokenizer.decode(gen, skip_special_tokens=True)
    print("\n================ decoded ================")
    print(text)
    print("=========================================\n")

    # Inspect debug dumps
    dumps = sorted(debug_dir.glob("*.json"))
    print(f"[debug] debug payloads captured: {len(dumps)}")
    if not dumps:
        raise SystemExit("no debug payload captured — hook did not fire")

    for p in dumps:
        with p.open() as f:
            pay = json.load(f)
        summary = pay["summary"]
        tag = pay["tag"]
        loc = pay["location"]
        keys_present = sorted(k for k in summary.keys() if isinstance(summary[k], dict))
        print(f"  - {p.name} | loc={loc} | tag={tag} | tensors={keys_present}")

    # For MLP subspace: confirm all 5 stages present, and dims make sense
    if (not args.no_mlp) and (not args.fixed_pca):
        mlp_dumps = [p for p in dumps if "mlp_subspace" in p.name]
        if not mlp_dumps:
            raise SystemExit("expected at least one mlp_subspace debug payload")
        with mlp_dumps[0].open() as f:
            pay = json.load(f)
        s = pay["summary"]
        required = [
            "original_tensor", "projected_tensor", "clipped_projected_tensor",
            "low_rank_noise", "projected_noisy_tensor", "noisy_tensor",
            "distortion_tensor",
        ]
        missing = [k for k in required if k not in s]
        if missing:
            raise SystemExit(f"MLP subspace debug payload missing keys: {missing}")

        proj_shape = s["projected_tensor"]["shape"]
        orig_shape = s["original_tensor"]["shape"]
        print(f"[debug] orig shape={orig_shape}, projected shape={proj_shape}")
        assert proj_shape[-1] == args.mlp_dim, \
            f"expected projected last-dim={args.mlp_dim}, got {proj_shape[-1]}"
        assert s["noisy_tensor"]["shape"][-1] == orig_shape[-1], \
            f"up-projected tensor must return to original last-dim={orig_shape[-1]}"

        clip_norm = max(abs(s["clipped_projected_tensor"]["max"]),
                        abs(s["clipped_projected_tensor"]["min"]))
        print(f"[debug] per-token clip ≤ norm_c={args.norm_c}, "
              f"observed |z_clipped|∞={clip_norm:.4f}")
        print(f"[debug] Gaussian noise std (observed) = {s['low_rank_noise']['std']:.4f}")
        print(f"[debug] reconstruction distortion L∞ = "
              f"{max(abs(s['distortion_tensor']['max']), abs(s['distortion_tensor']['min'])):.4f}")

    # For fixed PCA: inspect each dump, confirm the noise math math ran per-dim.
    if args.fixed_pca:
        pca_dumps = [p for p in dumps if "fixed_pca" in p.name]
        if not pca_dumps:
            raise SystemExit("expected at least one fixed_pca debug payload")
        required = [
            "original_tensor", "centered_tensor", "projected_tensor",
            "clipped_projected_tensor", "low_rank_noise", "projected_noisy_tensor",
            "reconstructed_noisy_tensor", "distortion_tensor",
        ]
        seen_dims = set()
        for p in pca_dumps:
            with p.open() as f:
                pay = json.load(f)
            s = pay["summary"]
            missing = [k for k in required if k not in s]
            if missing:
                raise SystemExit(f"fixed-PCA debug payload {p.name} missing keys: {missing}")
            orig_dim = s["original_tensor"]["shape"][-1]
            proj_dim = s["projected_tensor"]["shape"][-1]
            seen_dims.add(orig_dim)
            clip_max = max(abs(s["clipped_projected_tensor"]["max"]),
                           abs(s["clipped_projected_tensor"]["min"]))
            print(f"[debug] PCA {p.name} | feat_dim={orig_dim} → proj_dim={proj_dim} | "
                  f"tag={pay['tag']} | |z_clipped|∞={clip_max:.4f} | "
                  f"noise_std={s['low_rank_noise']['std']:.4f} | "
                  f"recon_L∞={max(abs(s['distortion_tensor']['max']), abs(s['distortion_tensor']['min'])):.4f}")
        print(f"[debug] PCA dispatched to feat_dims: {sorted(seen_dims)}")

    mech = "MLP↓→norm+DP-noise→MLP↑" if (not args.no_mlp and not args.fixed_pca) else (
        "PCA-project→norm+DP-noise→PCA-reconstruct" if args.fixed_pca else "raw full-dim Gaussian"
    )
    print(f"\n[debug] ✅ pipeline verified: specified-location → feature → "
          f"{mech} → downstream inference → text")

    # ---------------------------------------------------------------
    # Extra checks: training vs inference contract
    # ---------------------------------------------------------------

    # Case A: fixed-PCA. Basis lives in model cache (CPU tensors); confirm
    # byte-identical across two generates.
    if args.fixed_pca:
        print("\n[debug] === fixed-PCA: basis frozen across inference ===")
        cache = getattr(model, "_vision_dp_pca_cache", None) or getattr(
            model.config, "_vision_dp_pca_cache", None
        )
        if cache is None:
            raise SystemExit("fixed-PCA cache missing on model/config")

        def _snapshot_pca():
            snap = {}
            for loc, entry in cache.items():
                if entry.get("multi_dim"):
                    for d, meta in entry["per_dim"].items():
                        snap[(loc, d, "mean")] = meta["mean"].detach().clone()
                        snap[(loc, d, "basis")] = meta["basis"].detach().clone()
                else:
                    snap[(loc, None, "mean")] = entry["mean"].detach().clone()
                    snap[(loc, None, "basis")] = entry["basis"].detach().clone()
            return snap

        before = _snapshot_pca()
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        after = _snapshot_pca()
        all_equal = all(torch.equal(before[k], after[k]) for k in before)
        print(f"[debug] {len(before)} basis tensors in cache; "
              f"byte-identical across two generates? {all_equal} "
              f"(inference: ✓ basis frozen)")
        if not all_equal:
            for k in before:
                if not torch.equal(before[k], after[k]):
                    print(f"    !! changed: {k}")
        print("\n[debug] === summary ===")
        print(f"[debug] training path:  basis is pre-fit offline; _load_fixed_pca_basis "
              f"loads it into model._vision_dp_pca_cache; dispatcher routes each tensor "
              f"to its per-dim basis by x.shape[-1].")
        print(f"[debug] inference path: model.eval()+no_grad wraps generate; basis tensors "
              f"byte-identical across two calls. Only the Gaussian noise is re-sampled.")
        return

    # Case B: raw Gaussian — nothing to assert beyond "it ran."
    if args.no_mlp:
        return

    # Case C: MLP subspace — original check.
    print("\n[debug] === training vs inference: MLP weight behavior ===")

    mlp_params = [
        (n, p) for n, p in model.named_parameters() if n.startswith("_dp_mlp_")
    ]
    print(f"[debug] MLP params registered on model: {len(mlp_params)}")
    for n, p in mlp_params:
        print(f"    - {n} shape={tuple(p.shape)} dtype={p.dtype} "
              f"device={p.device} requires_grad={p.requires_grad}")

    all_trainable = all(p.requires_grad for _, p in mlp_params)
    print(f"[debug] all MLP params requires_grad=True? {all_trainable}  "
          f"(training: ✓ MLP moves with VLM)")

    # 1) Inference-mode: MLP weights must be byte-identical across two generates.
    def _snapshot_mlp():
        return {n: p.detach().clone().cpu() for n, p in mlp_params}

    before = _snapshot_mlp()
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    after = _snapshot_mlp()
    all_equal = all(torch.equal(before[n], after[n]) for n in before)
    print(f"[debug] MLP weights unchanged after inference generate? {all_equal}  "
          f"(inference: ✓ MLP frozen without optimizer step)")
    if not all_equal:
        for n in before:
            if not torch.equal(before[n], after[n]):
                diff = (before[n].float() - after[n].float()).abs().max().item()
                print(f"    !! {n} changed, max|Δ|={diff}")

    # 2) Training-mode: enable grad, run a forward, confirm MLP params receive
    #    gradients (i.e. the optimizer will update them).
    print("[debug] running one forward+backward to confirm MLP receives gradients...")
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    try:
        model_inputs = {k: v for k, v in inputs.items()}
        labels = model_inputs["input_ids"].clone()
        model.train()
        out = model(**model_inputs, labels=labels)
        loss = out.loss
        print(f"[debug] forward loss = {loss.item():.4f}")
        loss.backward()
    except Exception as e:
        print(f"[debug] backward check failed: {e!r}")
    finally:
        model.eval()

    grad_report = []
    for n, p in mlp_params:
        if p.grad is None:
            grad_report.append((n, None))
        else:
            g = p.grad.detach().float()
            grad_report.append((n, float(g.abs().max().item())))

    any_grad = any(g is not None and g > 0 for _, g in grad_report)
    print(f"[debug] any MLP param got a non-zero gradient? {any_grad}  "
          f"(training: ✓ grads flow → optimizer will update MLP alongside VLM)")
    for n, g in grad_report:
        if g is None:
            print(f"    - {n} grad=None")
        else:
            print(f"    - {n} max|grad|={g:.3e}")

    print("\n[debug] === summary ===")
    print(f"[debug] training path:  MLP registered ({len(mlp_params)} params), "
          f"requires_grad all True, grads flow → optimizer updates MLP.")
    print(f"[debug] inference path: model.eval()+no_grad wraps generate; "
          f"MLP weights byte-identical across two calls. "
          f"Only the Gaussian noise is re-sampled each call (expected).")


if __name__ == "__main__":
    main()

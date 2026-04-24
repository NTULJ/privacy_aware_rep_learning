"""Fixed PCA DP noise mechanism.

Pipeline: x -> (x - mean) -> project to r-dim via basis -> per-token clip
to norm_c -> + Gaussian noise -> reconstruct back to feat_dim -> + mean.

Two on-disk layouts per location:
  (a) Single-dim (legacy):   <pca_basis_dir>/<safe_loc>.pt
  (b) Multi-dim:             <pca_basis_dir>/<safe_loc>__dim<N>.pt
      — one file per feature dim encountered at the same hook location.
      Needed for locations where the hook sees multiple last-dims, e.g.
      vit_output (4096 last_hidden_state + 1152 deepstack_features) and
      vit_hidden_k>8 (1152 block output + 4096 post-merger deepstack backfill).

Schema per file (byte-identical between train-time fit and inference load):
    {
      "mean":  [1, dim]  fp32,
      "basis": [rank, dim] fp32,
      "rank": int,
      "effective_rank": int,
      "dim": int,
      "explained_variance_ratio": float,
    }

Cache entry shape populated into model._vision_dp_pca_cache (not config; tensors are not JSON-serializable):
  - single-dim: {"mean", "basis", "rank", "effective_rank", "dim",
                 "path", "explained_variance_ratio"}
  - multi-dim:  {"multi_dim": True,
                 "per_dim": {N: {same keys as single-dim}, ...}}
"""
from __future__ import annotations

import glob as _glob
import os

import torch

from ._common import _safe_loc_name, _save_debug_payload


def _load_single_basis_file(path: str) -> dict:
    """Load one basis .pt file; validate; return the standard cache-entry dict."""
    payload = torch.load(path, map_location="cpu")
    if "mean" not in payload or "basis" not in payload:
        raise RuntimeError(
            f"[vision_dp_everywhere] PCA basis file missing mean/basis keys: {path}"
        )

    mean = payload["mean"].float().cpu().reshape(1, -1)
    basis = payload["basis"].float().cpu()

    if basis.ndim != 2:
        raise RuntimeError(
            f"[vision_dp_everywhere] PCA basis must be 2D [rank, dim], got {list(basis.shape)} at {path}"
        )
    if mean.ndim != 2 or mean.shape[0] != 1:
        raise RuntimeError(
            f"[vision_dp_everywhere] PCA mean must be [1, dim], got {list(mean.shape)} at {path}"
        )
    if mean.shape[1] != basis.shape[1]:
        raise RuntimeError(
            f"[vision_dp_everywhere] PCA mean/basis dim mismatch at {path}: "
            f"mean_dim={mean.shape[1]}, basis_dim={basis.shape[1]}"
        )

    return {
        "mean": mean,
        "basis": basis,
        "rank": int(payload.get("rank", basis.shape[0])),
        "effective_rank": int(payload.get("effective_rank", basis.shape[0])),
        "dim": int(payload.get("dim", basis.shape[1])),
        "path": path,
        "explained_variance_ratio": float(payload.get("explained_variance_ratio", -1.0)),
    }


def _load_fixed_pca_basis(config, locations):
    """Load PCA bases for every requested location. Returns the cache dict."""
    basis_dir = getattr(config, "vision_dp_pca_basis_dir", None)
    if basis_dir is None or str(basis_dir).strip() == "":
        raise RuntimeError(
            "[vision_dp_everywhere] vision_dp_pca_basis_dir is required when vision_dp_use_fixed_pca=True"
        )

    cache = {}
    for loc in locations:
        safe = _safe_loc_name(loc)
        single_path = os.path.join(basis_dir, f"{safe}.pt")
        multi_paths = sorted(_glob.glob(os.path.join(basis_dir, f"{safe}__dim*.pt")))

        has_single = os.path.exists(single_path)
        has_multi = len(multi_paths) > 0

        if has_single and has_multi:
            raise RuntimeError(
                f"[vision_dp_everywhere] both single-dim ({single_path}) and multi-dim "
                f"({[os.path.basename(p) for p in multi_paths]}) basis files exist for "
                f"location={loc}. Remove one layout — mixing is ambiguous."
            )

        if has_single:
            # Single-dim (legacy) layout. Keep the historical restriction: a single
            # basis can't serve the mixed-dim locations (vit_input/vit_output).
            if loc in {"vit_input", "vit_output"}:
                raise RuntimeError(
                    f"[vision_dp_everywhere] single-dim basis is not supported for location={loc}; "
                    f"use the multi-dim layout ({safe}__dim<N>.pt) or pick a single-dim location."
                )
            cache[loc] = _load_single_basis_file(single_path)
            continue

        if has_multi:
            per_dim = {}
            for p in multi_paths:
                entry = _load_single_basis_file(p)
                feat_dim = int(entry["dim"])
                if feat_dim in per_dim:
                    raise RuntimeError(
                        f"[vision_dp_everywhere] duplicate multi-dim basis for dim={feat_dim} "
                        f"at {loc}: {per_dim[feat_dim]['path']} vs {p}"
                    )
                per_dim[feat_dim] = entry
            cache[loc] = {"multi_dim": True, "per_dim": per_dim}
            continue

        raise RuntimeError(
            f"[vision_dp_everywhere] no PCA basis file found for location={loc}; "
            f"looked for {single_path} and {safe}__dim*.pt under {basis_dir}"
        )

    return cache


def _pca_project_clip_noise(
    embeddings: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    noise_factor: float,
    norm_c: float,
):
    """Core centered-PCA noise math shared by single-dim and multi-dim paths."""
    orig_dtype = embeddings.dtype
    orig_shape = embeddings.shape

    if embeddings.ndim < 2:
        raise RuntimeError(
            f"[vision_dp_everywhere] fixed PCA expects tensor ndim >= 2, got shape={list(orig_shape)}"
        )

    x = embeddings.float()
    dim = x.shape[-1]
    x2 = x.reshape(-1, dim)

    mean = mean.to(device=x.device, dtype=torch.float32)
    basis = basis.to(device=x.device, dtype=torch.float32)

    if x2.shape[1] != basis.shape[1]:
        raise RuntimeError(
            f"[vision_dp_everywhere] PCA dim mismatch: input_dim={x2.shape[1]}, basis_dim={basis.shape[1]}"
        )

    x_centered = x2 - mean
    z = torch.matmul(x_centered, basis.t())  # [N, r]

    token_norm = torch.norm(z, dim=-1)
    clip_coef = norm_c / (token_norm + 1e-6)
    clip_coef = torch.clamp(clip_coef, max=1.0)
    z_clipped = z * clip_coef.unsqueeze(-1)

    noise = noise_factor * torch.randn_like(z_clipped)
    z_noisy = z_clipped + noise

    x_recon = torch.matmul(z_noisy, basis) + mean
    x_recon = x_recon.reshape(orig_shape)

    debug = {
        "original_tensor": embeddings,
        "centered_tensor": x_centered.reshape(orig_shape),
        "projected_tensor": z,
        "clipped_projected_tensor": z_clipped,
        "low_rank_noise": noise,
        "projected_noisy_tensor": z_noisy,
        "reconstructed_noisy_tensor": x_recon,
        "distortion_tensor": x_recon - embeddings.float(),
        "noise_factor": float(noise_factor),
        "norm_c": float(norm_c),
    }
    return x_recon.to(orig_dtype), debug


def _add_noise_with_fixed_pca(
    embeddings: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    noise_factor: float = 0.5,
    norm_c: float = 1.0,
    add_noise: bool = True,
    config=None,
    location: str = "",
    debug_tag: str = "fixed_pca",
):
    if not add_noise or noise_factor <= 0:
        return embeddings

    if not torch.is_tensor(embeddings):
        return embeddings

    if not embeddings.is_floating_point():
        return embeddings

    x_recon, debug = _pca_project_clip_noise(
        embeddings=embeddings,
        mean=mean,
        basis=basis,
        noise_factor=noise_factor,
        norm_c=norm_c,
    )

    if config is not None:
        _save_debug_payload(
            config=config,
            location=location,
            tag=debug_tag,
            payload=debug,
        )

    return x_recon

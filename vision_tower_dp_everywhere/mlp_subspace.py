"""MLP-subspace DP noise mechanism.

Pipeline: x (feat_dim) -> mlp_down -> z (hidden_dim) -> per-token clip to
norm_c -> + Gaussian noise -> mlp_up -> y (feat_dim).

The MLP modules themselves live on the top-level model as
`model._dp_mlp_down` / `model._dp_mlp_up` so that:
  - FSDP wraps them as sub-modules and includes them in state_dict;
  - optimizer sees them via `model.named_parameters()` (a name-prefix match
    in the FSDP engine puts them in the weight_decay=0 group — see
    verl/verl/workers/engine/fsdp/transformer_impl.py:386-419).

Registration lives in patch._install_hook_if_needed (it needs to know the
vision tower's hidden_dim). This module only provides build + noise helpers.
"""
from __future__ import annotations

import torch

from ._common import _save_debug_payload


def _create_mlp_subspace(input_dim: int, hidden_dim: int, device, dtype):
    down_proj = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
    ).to(device=device, dtype=dtype)

    up_proj = torch.nn.Linear(hidden_dim, input_dim).to(device=device, dtype=dtype)

    return down_proj, up_proj


def _add_noise_with_mlp_subspace(
    embeddings: torch.Tensor,
    mlp_down: torch.nn.Module,
    mlp_up: torch.nn.Module,
    noise_factor: float = 0.5,
    norm_c: float = 1.0,
    config=None,
    location: str = "",
):
    if not torch.is_tensor(embeddings):
        return embeddings
    if not embeddings.is_floating_point():
        return embeddings

    orig_dtype = embeddings.dtype

    # Match the MLP weight dtype — model weights are typically bfloat16,
    # so upcasting to float32 would break the linear (`mat1/mat2 dtype mismatch`).
    mlp_dtype = next(mlp_down.parameters()).dtype
    x = embeddings.to(mlp_dtype)

    z = mlp_down(x)

    token_norm = torch.norm(z, dim=-1, keepdim=True)
    clip_coef = torch.clamp(norm_c / (token_norm + 1e-10), max=1.0)
    z_clipped = z * clip_coef

    noise = noise_factor * torch.randn_like(z_clipped)
    z_noisy = z_clipped + noise

    x_noisy = mlp_up(z_noisy)

    if config is not None:
        _save_debug_payload(
            config=config,
            location=location,
            tag="mlp_subspace",
            payload={
                "original_tensor": embeddings,
                "projected_tensor": z,
                "clipped_projected_tensor": z_clipped,
                "low_rank_noise": noise,
                "projected_noisy_tensor": z_noisy,
                "noisy_tensor": x_noisy,
                "distortion_tensor": x_noisy - embeddings.float(),
                "noise_factor": float(noise_factor),
                "norm_c": float(norm_c),
            },
        )

    return x_noisy.to(orig_dtype)

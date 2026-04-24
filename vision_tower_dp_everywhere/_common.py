"""Shared helpers for dp/patch.py and mechanism sub-modules.

Kept small on purpose: the only things here are pieces used by BOTH noise
mechanisms (MLP subspace, fixed PCA) and by the orchestrator (patch.py).
Anything mechanism-specific belongs in mlp_subspace.py / fixed_pca.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def _rank0_print(msg: str):
    if int(os.environ.get("RANK", "0")) == 0:
        print(msg, flush=True)


def _safe_loc_name(location: str) -> str:
    return location.replace("/", "__")


# =========================
# Debug helpers
# =========================
def _debug_enabled(config) -> bool:
    return bool(getattr(config, "vision_dp_debug", True))


def _debug_dir(config) -> str:
    return str(getattr(config, "vision_dp_debug_dir", "./vision_dp_debug"))


def _debug_max_tokens(config) -> int:
    return int(getattr(config, "vision_dp_debug_max_tokens", 4))


def _debug_max_dims(config) -> int:
    return int(getattr(config, "vision_dp_debug_max_dims", 8))


def _debug_save_full(config) -> bool:
    return bool(getattr(config, "vision_dp_debug_save_full_tensors", False))


def _debug_max_dump(config) -> int:
    return int(getattr(config, "vision_dp_debug_max_dump", 1))


def _debug_counter(config) -> int:
    return int(getattr(config, "_vision_dp_debug_counter", 0))


def _inc_debug_counter(config):
    setattr(config, "_vision_dp_debug_counter", _debug_counter(config) + 1)


def _preview_tensor(x: torch.Tensor, max_tokens: int = 4, max_dims: int = 8):
    x = x.detach().float().cpu()
    if x.ndim == 3:
        return x[0, :max_tokens, :max_dims]
    if x.ndim == 2:
        return x[:max_tokens, :max_dims]
    if x.ndim == 1:
        return x[:max_dims]
    return x.reshape(-1)[: max_tokens * max_dims]


def _tensor_summary(x: torch.Tensor):
    x = x.detach().float().cpu()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "fro_norm": float(torch.linalg.vector_norm(x.reshape(-1), ord=2).item()),
    }


def _save_debug_payload(config, location: str, tag: str, payload: dict):
    if not _debug_enabled(config):
        return
    if int(os.environ.get("RANK", "0")) != 0:
        return

    counter = _debug_counter(config)
    if counter >= _debug_max_dump(config):
        return

    out_dir = Path(_debug_dir(config))
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_loc = _safe_loc_name(location)
    stem = f"{counter:04d}_{safe_loc}_{tag}"

    max_tokens = _debug_max_tokens(config)
    max_dims = _debug_max_dims(config)

    preview = {}
    summary = {}
    full_tensors = {}

    for k, v in payload.items():
        if torch.is_tensor(v):
            summary[k] = _tensor_summary(v)
            preview[k] = _preview_tensor(v, max_tokens=max_tokens, max_dims=max_dims).tolist()
            if _debug_save_full(config):
                full_tensors[k] = v.detach().float().cpu()
        else:
            summary[k] = v

    json_path = out_dir / f"{stem}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "location": location,
                "tag": tag,
                "summary": summary,
                "preview": preview,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if _debug_save_full(config) and full_tensors:
        torch.save(full_tensors, out_dir / f"{stem}.pt")

    _rank0_print(f"[vision_tower_dp_everywhere] saved debug payload -> {json_path}")
    _inc_debug_counter(config)

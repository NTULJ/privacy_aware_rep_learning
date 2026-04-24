import os

import torch

print("[vision_tower_dp_everywhere] patch.py imported", flush=True)

from .dpforward import add_noise, get_noise_multiplier, _max_norm_clip
from ._common import (
    _rank0_print,
    _safe_loc_name,
    _save_debug_payload,
    _debug_enabled,
    _debug_dir,
)
from .mlp_subspace import _create_mlp_subspace, _add_noise_with_mlp_subspace
from .fixed_pca import _load_fixed_pca_basis, _add_noise_with_fixed_pca


# =========================
# Config resolution helpers
# =========================
def _locate_vision_tower(model):
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual, "model.visual"
    if hasattr(model, "visual"):
        return model.visual, "visual"
    if hasattr(model, "vision_tower"):
        return model.vision_tower, "vision_tower"
    return None, None


def _resolve_noise_factor(config) -> float:
    eps = float(getattr(config, "vision_dp_epsilon", -1.0))
    if eps == -1.0:
        return 0.0

    delta = float(getattr(config, "vision_dp_delta", 1e-5))
    local_dp = bool(getattr(config, "vision_dp_local_dp", True))
    noise_type = str(getattr(config, "vision_dp_noise_type", "aGM"))
    # Calibrate sensitivity from runtime norm bound C with upper bound 2C.
    norm_c = float(getattr(config, "vision_dp_norm_c", 1.0))
    sensitivity = max(2.0 * norm_c, 1e-12)

    noise_factor = get_noise_multiplier(
        eps=eps,
        delta=delta,
        batch_size=1,
        dataset_size=50000,
        epoch=3,
        local_dp=local_dp,
        noise_type=noise_type,
        sensitivity=sensitivity,
    )
    setattr(config, "_vision_dp_resolved_sensitivity", float(sensitivity))
    return float(noise_factor)


def _resolve_mlp_subspace_config(config):
    use_mlp = bool(getattr(config, "vision_dp_use_mlp_subspace", False))
    use_pca = bool(getattr(config, "vision_dp_use_fixed_pca", False))

    if use_mlp and use_pca:
        raise RuntimeError(
            "[vision_dp_everywhere] Cannot use both MLP subspace and PCA subspace. "
            "Please set either vision_dp_use_mlp_subspace=True OR vision_dp_use_fixed_pca=True, not both."
        )

    hidden_dim = int(getattr(config, "vision_dp_mlp_hidden_dim", 128))
    return use_mlp, hidden_dim


def _resolve_deepstack_config(config):
    ds_indexes = getattr(config, "vision_dp_deepstack_indexes", [8, 16, 24])
    if isinstance(ds_indexes, str):
        ds_indexes = [int(x.strip()) for x in ds_indexes.split(",")]
    return ds_indexes


def _discover_deepstack_indexes(vision_tower, model) -> list[int]:
    """Discover the real deepstack capture layer indexes from the live model.

    Priority: live vision_tower attribute > vision_tower.config > model.config.vision_config.
    Falls back to the DP-side config, then empty list.
    """
    # 1. Live vision tower attribute (most authoritative)
    idx = getattr(vision_tower, "deepstack_visual_indexes", None)
    if idx is not None:
        return sorted([int(i) for i in idx])

    # 2. vision_tower.config
    vt_config = getattr(vision_tower, "config", None)
    if vt_config is not None:
        idx = getattr(vt_config, "deepstack_visual_indexes", None)
        if idx is not None:
            return sorted([int(i) for i in idx])

    # 3. model.config.vision_config
    model_config = getattr(model, "config", None)
    if model_config is not None:
        vis_config = getattr(model_config, "vision_config", None)
        if vis_config is not None:
            idx = getattr(vis_config, "deepstack_visual_indexes", None)
            if idx is not None:
                return sorted([int(i) for i in idx])

    # 4. DP-side override (least authoritative)
    config = getattr(model, "config", None)
    if config is not None:
        dp_idx = _resolve_deepstack_config(config)
        if dp_idx:
            return sorted([int(i) for i in dp_idx])

    return []


def _get_vision_tower_hidden_dim(vision_tower):
    if hasattr(vision_tower, "config"):
        config = vision_tower.config
        if hasattr(config, "vision_config"):
            vision_config = config.vision_config
            if hasattr(vision_config, "hidden_size"):
                return vision_config.hidden_size
        if hasattr(config, "hidden_size"):
            return config.hidden_size

    if hasattr(vision_tower, "embeddings") and hasattr(vision_tower.embeddings, "patch_embedding"):
        proj = getattr(vision_tower.embeddings.patch_embedding, "proj", None)
        if proj is not None:
            return proj.out_features
    if hasattr(vision_tower, "proj"):
        return vision_tower.proj.out_features
    if hasattr(vision_tower, "head"):
        return vision_tower.head.in_features

    return None


def _parse_noise_locations(config):
    locations = getattr(config, "vision_dp_noise_location", "vit_output")
    if isinstance(locations, str):
        return [locations]
    if isinstance(locations, list):
        return locations
    return ["vit_output"]


def _get_vit_layer_count(vision_tower):
    if hasattr(vision_tower, "blocks"):
        return len(vision_tower.blocks)
    if hasattr(vision_tower, "encoder") and hasattr(vision_tower.encoder, "layers"):
        return len(vision_tower.encoder.layers)
    if hasattr(vision_tower, "depth"):
        return vision_tower.depth
    return None



# (Helpers _safe_loc_name + _debug_* + _save_debug_payload live in dp/_common.py
# and are imported at the top of this file.)


# (Fixed PCA loader lives in dp/fixed_pca.py as _load_fixed_pca_basis and is
# imported at the top of this file.)


# =========================
# Noise mechanisms
# =========================
def _add_noise_with_debug_original(
    embeddings: torch.Tensor,
    noise_factor: float,
    norm_c: float,
    config,
    location: str,
):
    if not torch.is_tensor(embeddings):
        return embeddings
    if not embeddings.is_floating_point():
        return embeddings

    original = embeddings
    clipped = _max_norm_clip(embeddings, norm_c)
    noise = noise_factor * torch.randn(
        size=clipped.shape,
        dtype=clipped.dtype,
        device=clipped.device,
    )
    noisy = clipped + noise

    _save_debug_payload(
        config=config,
        location=location,
        tag="original_space",
        payload={
            "original_tensor": original,
            "clipped_signal": clipped,
            "noise_matrix": noise,
            "noisy_tensor": noisy,
            "distortion_tensor": noisy.float() - original.float(),
            "noise_factor": float(noise_factor),
            "norm_c": float(norm_c),
        },
    )
    return noisy


def _create_mlp_subspace(input_dim: int, hidden_dim: int, device, dtype):
    # Re-exported from dp/mlp_subspace.py (needed by _install_hook_if_needed
    # for the initial MLP registration).
    from .mlp_subspace import _create_mlp_subspace as _impl
    return _impl(input_dim, hidden_dim, device, dtype)


# (MLP noise mechanism lives in dp/mlp_subspace.py as _add_noise_with_mlp_subspace.)
# (Fixed-PCA noise math lives in dp/fixed_pca.py as _pca_project_clip_noise
#  and _add_noise_with_fixed_pca.)


# =========================
# Unified noise application
# =========================
def _apply_dp_noise(hidden_states, config, model=None, location=""):
    """Apply DP noise to hidden_states.

    Args:
        hidden_states: tensor to add noise to
        config: model config with DP parameters
        model: top-level model (needed to access model._dp_mlp_down/up)
        location: noise location string for debugging
    """
    if not torch.is_tensor(hidden_states):
        return hidden_states

    if not hidden_states.is_floating_point():
        return hidden_states

    noise_factor = float(getattr(config, "_vision_dp_resolved_noise_factor", 0.0))
    norm_c = float(getattr(config, "vision_dp_norm_c", 1.0))

    if noise_factor <= 0:
        return hidden_states

    use_mlp_subspace, _ = _resolve_mlp_subspace_config(config)
    use_fixed_pca = bool(getattr(config, "vision_dp_use_fixed_pca", False))

    if use_mlp_subspace:
        # *** 关键改动: MLP 从 model 取，不再从 config 取 ***
        # 原因: setattr(config, "_vision_dp_mlp_down", mlp_down) 把 nn.Module 挂到 config 上，
        #        导致 HF 保存 config 时 JSON 序列化崩溃 (TypeError: Object of type Sequential is not JSON serializable)
        mlp_down = getattr(model, "_dp_mlp_down", None)
        mlp_up = getattr(model, "_dp_mlp_up", None)
        if mlp_down is None or mlp_up is None:
            raise RuntimeError(
                "[vision_dp_everywhere] MLP subspace enabled but MLP not found on model. "
                "Ensure model._dp_mlp_down and model._dp_mlp_up are set."
            )

        # MLP input_dim must match the feature dim at the hook location. vit_output
        # is rejected upfront in _install_hook_if_needed because it mixes 4096-dim
        # last_hidden_state with 1152-dim deepstack_features. All remaining
        # MLP-compatible locations (vit_input / vit_after_pos_embed / vit_hidden_k)
        # share vision_config.hidden_size, so the pre-built MLP always matches.
        feat_dim = int(hidden_states.shape[-1])
        current_in_dim = (
            mlp_down[0].in_features if isinstance(mlp_down, torch.nn.Sequential)
            else mlp_down.in_features
        )
        if current_in_dim != feat_dim:
            raise RuntimeError(
                f"[vision_dp_everywhere] MLP input_dim={current_in_dim} does not match "
                f"feature dim={feat_dim} at location={location}. This should not happen "
                f"if vit_output is blocked — please file a bug with the location."
            )

        return _add_noise_with_mlp_subspace(
            embeddings=hidden_states,
            mlp_down=mlp_down,
            mlp_up=mlp_up,
            noise_factor=noise_factor,
            norm_c=norm_c,
            config=config,
            location=location,
        )

    if not use_fixed_pca:
        return _add_noise_with_debug_original(
            embeddings=hidden_states,
            noise_factor=noise_factor,
            norm_c=norm_c,
            config=config,
            location=location,
        )

    # PCA cache holds torch.Tensor mean/basis — must NOT live on config or HF
    # save_pretrained(config) will JSON-serialize the whole config dict and crash.
    pca_cache = getattr(model, "_vision_dp_pca_cache", None) if model is not None else None
    if pca_cache is None:
        pca_cache = getattr(config, "_vision_dp_pca_cache", None)
    if pca_cache is None:
        raise RuntimeError(
            "[vision_dp_everywhere] vision_dp_use_fixed_pca=True but no PCA cache is loaded"
        )
    if location not in pca_cache:
        raise RuntimeError(
            f"[vision_dp_everywhere] no PCA basis loaded for location={location}"
        )

    entry = pca_cache[location]
    if entry.get("multi_dim"):
        feat_dim = int(hidden_states.shape[-1])
        per_dim = entry["per_dim"]
        if feat_dim not in per_dim:
            raise RuntimeError(
                f"[vision_dp_everywhere] no PCA basis for dim={feat_dim} at location={location}; "
                f"available dims: {sorted(per_dim.keys())}. "
                f"The offline fit tool must produce {_safe_loc_name(location)}__dim{feat_dim}.pt."
            )
        meta = per_dim[feat_dim]
        debug_tag = f"fixed_pca_multi_dim_{feat_dim}"
    else:
        meta = entry
        debug_tag = "fixed_pca"

    return _add_noise_with_fixed_pca(
        embeddings=hidden_states,
        mean=meta["mean"],
        basis=meta["basis"],
        noise_factor=noise_factor,
        norm_c=norm_c,
        add_noise=True,
        config=config,
        location=location,
        debug_tag=debug_tag,
    )


# =========================
# Hook builders
# =========================
def _make_hook(config, location, model):
    """Build a forward hook that applies DP noise.

    Args:
        config: model config
        location: noise location string (e.g. "vit_hidden_8", "vit_output")
        model: top-level model (for accessing _dp_mlp_down/up)
    """

    def _noise_tree(x):
        """Recursively apply DP noise to tensors."""
        if torch.is_tensor(x):
            return _apply_dp_noise(x, config, model=model, location=location)
        if isinstance(x, tuple):
            return tuple(_noise_tree(v) for v in x)
        if isinstance(x, list):
            return [_noise_tree(v) for v in x]
        return x

    def hook(module, inputs, output):
        # Case 1: output is a plain tensor (e.g. individual ViT block)
        if torch.is_tensor(output):
            return _noise_tree(output)

        # Case 2: output is a tuple
        if isinstance(output, tuple):
            return tuple(_noise_tree(x) for x in output)

        # Case 3: output is a ModelOutput-like object
        touched = False

        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            output.last_hidden_state = _noise_tree(output.last_hidden_state)
            touched = True

        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            output.pooler_output = _noise_tree(output.pooler_output)
            touched = True

        if hasattr(output, "hidden_states") and output.hidden_states is not None:
            output.hidden_states = _noise_tree(output.hidden_states)
            touched = True

        if hasattr(output, "deepstack_features") and output.deepstack_features is not None:
            # For vit_hidden_* hooks on individual blocks, this code path is
            # typically never reached because individual blocks return plain
            # tensors, not objects with deepstack_features.
            #
            # IMPORTANT: vit_hidden_x noise only propagates naturally to
            # deepstack features whose capture layer >= x (because the noised
            # hidden_states flows into the deepstack merger at or after that
            # block).  Deepstack features captured at layers < x are already
            # materialised before this hook fires, so they BYPASS the noise.
            # That bypass is handled separately by a vision-tower-level
            # backfill hook registered in _install_hook_if_needed.
            #
            # For vit_output hooks on the entire vision_tower, deepstack_features
            # IS present in the output, and we noise them all here.
            output.deepstack_features = _noise_tree(output.deepstack_features)
            touched = True

        if not touched:
            raise RuntimeError(
                f"[vision_dp_everywhere] unsupported output type for {location}: {type(output)}"
            )

        return output

    return hook


def _make_pre_hook(config, location, model):
    """Build a forward pre-hook that applies DP noise to inputs.

    Args:
        config: model config
        location: noise location string
        model: top-level model (for accessing _dp_mlp_down/up)
    """

    def hook(module, input):
        if input is None or len(input) == 0:
            return None

        processed_input = []
        for x in input:
            if torch.is_tensor(x):
                noisy_x = _apply_dp_noise(x, config, model=model, location=location)
                processed_input.append(noisy_x)
            else:
                processed_input.append(x)
        return tuple(processed_input)

    return hook


def _make_deepstack_backfill_hook(config, location, model, deepstack_indexes: list[int], vit_hidden_idx: int):
    """Build a forward hook on the vision tower that applies DP noise to
    deepstack_features whose capture layer index is strictly less than
    ``vit_hidden_idx``.

    This closes the bypass where ``vit_hidden_x`` noised the hidden state
    *after* earlier deepstack features were already materialised.

    Args:
        config: model config
        location: the vit_hidden_x location string (for debug tagging)
        model: top-level model (for accessing _dp_mlp_down/up)
        deepstack_indexes: sorted list of deepstack capture layer indexes
        vit_hidden_idx: the x in vit_hidden_x
    """

    earlier_indexes = [idx for idx in deepstack_indexes if idx < vit_hidden_idx]

    def hook(module, inputs, output):
        if not hasattr(output, "deepstack_features") or output.deepstack_features is None:
            return output

        new_features = list(output.deepstack_features)
        for list_pos, capture_idx in enumerate(deepstack_indexes):
            if capture_idx in earlier_indexes and list_pos < len(new_features):
                new_features[list_pos] = _apply_dp_noise(
                    new_features[list_pos], config, model=model,
                    location=f"{location}/deepstack_{capture_idx}",
                )
        output.deepstack_features = new_features
        return output

    return hook, earlier_indexes


# =========================
# Patch installer
# =========================
def _install_hook_if_needed(model):
    if getattr(model, "_vision_dp_everywhere_hook_installed", False):
        return model

    config = getattr(model, "config", None)
    if config is None:
        _rank0_print("[vision_tower_dp_everywhere] model has no config, skip")
        return model

    enable = bool(getattr(config, "vision_dp_enable", True))
    if not enable:
        _rank0_print("[vision_tower_dp_everywhere] disabled, skip")
        return model

    noise_factor = _resolve_noise_factor(config)
    setattr(config, "_vision_dp_resolved_noise_factor", noise_factor)
    setattr(config, "_vision_dp_debug_counter", 0)

    if noise_factor <= 0:
        _rank0_print("[vision_tower_dp_everywhere] resolved noise_factor <= 0, skip")
        return model

    vision_tower, tower_name = _locate_vision_tower(model)
    if vision_tower is None:
        _rank0_print("[vision_tower_dp_everywhere] no vision tower found, skip")
        return model

    locations = _parse_noise_locations(config)
    num_layers = _get_vit_layer_count(vision_tower)

    # --- Fixed PCA setup ---
    use_fixed_pca = bool(getattr(config, "vision_dp_use_fixed_pca", False))
    if use_fixed_pca:
        # Auto-discover: if the user didn't set vision_dp_pca_basis_dir, look for
        # a `dp_pca_basis/` sibling of the HF model weights (mirrors the
        # dp_mlp_weights.pt auto-load convention below).
        basis_dir = getattr(config, "vision_dp_pca_basis_dir", None)
        if basis_dir is None or str(basis_dir).strip() == "":
            model_dir = getattr(model, "name_or_path", None)
            if model_dir:
                candidate = os.path.join(model_dir, "dp_pca_basis")
                if os.path.isdir(candidate):
                    setattr(config, "vision_dp_pca_basis_dir", candidate)
                    _rank0_print(
                        f"[vision_tower_dp_everywhere] auto-discovered PCA basis dir: {candidate}"
                    )

        pca_cache = _load_fixed_pca_basis(config, locations)
        # Store on model only (same rationale as MLP modules vs config).
        if hasattr(config, "_vision_dp_pca_cache"):
            try:
                delattr(config, "_vision_dp_pca_cache")
            except AttributeError:
                pass
        setattr(model, "_vision_dp_pca_cache", pca_cache)

        for loc, entry in pca_cache.items():
            if entry.get("multi_dim"):
                for feat_dim, meta in sorted(entry["per_dim"].items()):
                    _rank0_print(
                        f"[vision_tower_dp_everywhere] loaded fixed PCA (multi-dim) | "
                        f"loc={loc} | dim={feat_dim} | rank={meta['rank']} | "
                        f"effective_rank={meta['effective_rank']} | "
                        f"evr={meta['explained_variance_ratio']:.6f} | path={meta['path']}"
                    )
            else:
                _rank0_print(
                    f"[vision_tower_dp_everywhere] loaded fixed PCA | "
                    f"loc={loc} | rank={entry['rank']} | effective_rank={entry['effective_rank']} | "
                    f"dim={entry['dim']} | evr={entry['explained_variance_ratio']:.6f} | "
                    f"path={entry['path']}"
                )

    # --- MLP subspace setup ---
    use_mlp_subspace, mlp_hidden_dim = _resolve_mlp_subspace_config(config)
    if use_mlp_subspace:
        # MLP subspace requires every tensor flowing through the hook(s) to share
        # the same last-dim (= vision_config.hidden_size). Two gotchas:
        #
        # 1. vit_output carries both last_hidden_state (post-merger, out_hidden_size,
        #    e.g. 4096) and deepstack_features (hidden_size, e.g. 1152). Mixed dims.
        #
        # 2. For vit_hidden_k with k > any deepstack capture index, a "backfill"
        #    hook fires on the full vision tower to noise the earlier deepstack
        #    features. By that point those features have already been pushed
        #    through deepstack_merger_list and are out_hidden_size-dim, not
        #    hidden_size-dim. A single 1152-in MLP can't serve 1152 AND 4096.
        #
        # A single MLP cannot cover both dims, so we reject these configs upfront.
        ds_indexes = _discover_deepstack_indexes(vision_tower, model)
        min_ds = min(ds_indexes) if ds_indexes else None

        for loc in locations:
            if loc == "vit_output":
                raise RuntimeError(
                    "[vision_dp_everywhere] MLP subspace is not supported at vit_output: "
                    "last_hidden_state (post-merger) and deepstack_features have "
                    "different last-dims, a single MLP cannot handle both. "
                    "Use vit_input / vit_after_pos_embed / vit_hidden_k (k <= min deepstack idx) "
                    "instead, or switch to fixed PCA / raw full-dim Gaussian for vit_output."
                )
            if loc.startswith("vit_hidden_") and min_ds is not None:
                k = int(loc.split("_")[-1])
                if k > min_ds:
                    raise RuntimeError(
                        f"[vision_dp_everywhere] MLP subspace at {loc} is unsupported: "
                        f"deepstack captures at {ds_indexes} include layer(s) < {k}, "
                        f"which triggers the deepstack backfill hook on post-merger "
                        f"features (out_hidden_size-dim). A single MLP can't handle "
                        f"both the ViT-internal dim and the post-merger dim. "
                        f"Pick k <= {min_ds} (e.g. vit_hidden_{min_ds}), or use "
                        f"fixed PCA / raw full-dim Gaussian for deeper layers."
                    )
        if hasattr(model, '_dp_mlp_down') and hasattr(model, '_dp_mlp_up'):
            # FSDP 恢复训练：MLP 权重已由 FSDP 加载回来
            _rank0_print("[vision_tower_dp_everywhere] MLP subspace found, using loaded weights")
            mlp_down = model._dp_mlp_down
            mlp_up = model._dp_mlp_up
        else:
            # 首次训练 或 加载合并后的 HF 模型：需要创建 MLP
            hidden_dim = _get_vision_tower_hidden_dim(vision_tower)
            if hidden_dim is None:
                raise RuntimeError(
                    "[vision_dp_everywhere] cannot determine vision tower hidden dim for MLP"
                )
            mlp_down, mlp_up = _create_mlp_subspace(
                hidden_dim, mlp_hidden_dim,
                device=next(model.parameters()).device,
                dtype=next(model.parameters()).dtype
            )
            # Must use add_module (not plain setattr) so FSDP includes MLP
            # parameters in state_dict. Plain setattr makes them invisible
            # to model.state_dict() / model.named_parameters().
            model.add_module("_dp_mlp_down", mlp_down)
            model.add_module("_dp_mlp_up", mlp_up)
            _rank0_print(
                f"[vision_tower_dp_everywhere] created MLP subspace projector | "
                f"input_dim={hidden_dim} | hidden_dim={mlp_hidden_dim}"
            )

            # 尝试加载已训练的 MLP 权重（合并后的 HF 模型场景）
            model_dir = getattr(model, 'name_or_path', None)
            if model_dir:
                mlp_path = os.path.join(model_dir, "dp_mlp_weights.pt")
                if os.path.exists(mlp_path):
                    mlp_state = torch.load(mlp_path, map_location="cpu", weights_only=True)
                    down_state = {
                        k.replace("_dp_mlp_down.", ""): v
                        for k, v in mlp_state.items() if k.startswith("_dp_mlp_down.")
                    }
                    up_state = {
                        k.replace("_dp_mlp_up.", ""): v
                        for k, v in mlp_state.items() if k.startswith("_dp_mlp_up.")
                    }
                    mlp_down.load_state_dict(down_state)
                    mlp_up.load_state_dict(up_state)
                    _rank0_print("[vision_tower_dp_everywhere] loaded trained MLP weights from dp_mlp_weights.pt")

        # Mark MLP parameters as trainable + tag for weight_decay patching
        for p in mlp_down.parameters():
            p.requires_grad = True
            p._dp_mlp_param = True
        for p in mlp_up.parameters():
            p.requires_grad = True
            p._dp_mlp_param = True

        # *** 关键改动: 不再把 MLP 挂到 config 上 ***
        # 原来: setattr(config, "_vision_dp_mlp_down", mlp_down)
        # 原因: config.save_pretrained() 会 JSON 序列化 config，nn.Module 不可序列化
        # MLP 只存在 model._dp_mlp_down/up 上，hooks 通过 model 参数访问

    _rank0_print(
        f"[vision_tower_dp_everywhere] locations: {locations} | "
        f"vit_layers: {num_layers} | "
        f"noise_factor: {noise_factor:.6f} | "
        f"sensitivity: {getattr(config, '_vision_dp_resolved_sensitivity', None)} | "
        f"use_fixed_pca: {use_fixed_pca} | "
        f"debug={_debug_enabled(config)} | "
        f"debug_dir={_debug_dir(config)}"
    )

    # --- Discover deepstack capture indexes once ---
    deepstack_indexes = _discover_deepstack_indexes(vision_tower, model)
    if deepstack_indexes:
        _rank0_print(
            f"[vision_tower_dp_everywhere] detected deepstack_visual_indexes={deepstack_indexes}"
        )

    # --- Register hooks ---
    handles = []

    for loc in locations:
        try:
            if loc == "vit_input":
                if not hasattr(vision_tower, "patch_embed"):
                    raise RuntimeError(
                        f"[vision_dp_everywhere] vision_tower has no 'patch_embed' attribute, cannot register vit_input hook"
                    )
                patch_embed = vision_tower.patch_embed
                handle = patch_embed.register_forward_pre_hook(
                    _make_pre_hook(config, loc, model)
                )
                handles.append(handle)
                _rank0_print(f"  -> registered pre_hook on {tower_name}.patch_embed (vit_input)")

            elif loc == "vit_after_pos_embed":
                if not hasattr(vision_tower, "blocks") or len(vision_tower.blocks) == 0:
                    raise RuntimeError(
                        f"[vision_dp_everywhere] vision_tower has no blocks, cannot register vit_after_pos_embed"
                    )
                first_block = vision_tower.blocks[0]
                handle = first_block.register_forward_pre_hook(
                    _make_pre_hook(config, loc, model)
                )
                handles.append(handle)
                _rank0_print(f"  -> registered pre_hook on {tower_name}.blocks[0] (vit_after_pos_embed)")

            elif loc == "vit_output":
                handle = vision_tower.register_forward_hook(
                    _make_hook(config, loc, model)
                )
                handles.append(handle)
                _rank0_print(f"  -> registered forward_hook on {tower_name} (vit_output)")

            elif loc.startswith("vit_hidden_"):
                layer_idx = int(loc.split("_")[-1])
                if num_layers is None:
                    raise RuntimeError(
                        f"[vision_dp_everywhere] cannot detect vit layers, cannot register {loc}"
                    )
                if layer_idx >= num_layers:
                    raise RuntimeError(
                        f"[vision_dp_everywhere] layer {layer_idx} >= num_layers {num_layers}, invalid location: {loc}"
                    )

                target_block = vision_tower.blocks[layer_idx]
                handle = target_block.register_forward_hook(
                    _make_hook(config, loc, model)
                )
                handles.append(handle)
                _rank0_print(f"  -> registered forward_hook on {tower_name}.blocks[{layer_idx}] ({loc})")

                # Deepstack backfill: noise earlier deepstack features that
                # were materialised before this block ran.
                if deepstack_indexes:
                    backfill_hook, earlier = _make_deepstack_backfill_hook(
                        config, loc, model, deepstack_indexes, layer_idx,
                    )
                    if earlier:
                        ds_handle = vision_tower.register_forward_hook(backfill_hook)
                        handles.append(ds_handle)
                        _rank0_print(
                            f"  -> registered deepstack_backfill_hook on {tower_name} "
                            f"for earlier deepstack captures {earlier} ({loc})"
                        )

            else:
                raise RuntimeError(f"[vision_dp_everywhere] unknown location: {loc}")

        except Exception as e:
            raise RuntimeError(f"[vision_dp_everywhere] failed to register hook for {loc}: {e}")

    if not handles:
        raise RuntimeError("[vision_tower_dp_everywhere] no hooks registered, cannot continue")

    model._vision_dp_everywhere_hook_handles = handles
    model._vision_dp_everywhere_hook_installed = True

    _rank0_print(
        f"[vision_tower_dp_everywhere] installed {len(handles)} hook(s) | "
        f"epsilon={getattr(config, 'vision_dp_epsilon', None)} | "
        f"delta={getattr(config, 'vision_dp_delta', None)} | "
        f"local_dp={getattr(config, 'vision_dp_local_dp', None)} | "
        f"noise_type={getattr(config, 'vision_dp_noise_type', None)} | "
        f"norm_c={getattr(config, 'vision_dp_norm_c', None)} | "
        f"sensitivity={getattr(config, '_vision_dp_resolved_sensitivity', None)} | "
        f"noise_factor={getattr(config, '_vision_dp_resolved_noise_factor', None)} | "
        f"use_mlp_subspace={use_mlp_subspace} | "
        f"mlp_hidden_dim={mlp_hidden_dim if use_mlp_subspace else None} | "
        f"use_fixed_pca={getattr(config, 'vision_dp_use_fixed_pca', False)} | "
        f"pca_basis_dir={getattr(config, 'vision_dp_pca_basis_dir', '')} | "
        f"debug={_debug_enabled(config)}"
    )

    return model


def apply_patch():
    print("[vision_tower_dp_everywhere] apply_patch called", flush=True)
    from transformers.modeling_utils import PreTrainedModel

    if getattr(PreTrainedModel, "_vision_dp_everywhere_from_pretrained_patched", False):
        return

    old_from_pretrained = PreTrainedModel.from_pretrained.__func__

    def new_from_pretrained(cls, *args, **kwargs):
        model = old_from_pretrained(cls, *args, **kwargs)
        try:
            model = _install_hook_if_needed(model)
        except Exception as e:
            _rank0_print(f"[vision_tower_dp_everywhere] failed to install hook: {e}")
        return model

    PreTrainedModel.from_pretrained = classmethod(new_from_pretrained)
    PreTrainedModel._vision_dp_everywhere_from_pretrained_patched = True

    _rank0_print("[vision_tower_dp_everywhere] patched PreTrainedModel.from_pretrained")

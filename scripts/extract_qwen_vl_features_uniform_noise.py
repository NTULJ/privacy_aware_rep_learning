from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import extract_qwen_vl_features as base
import qwen3_vl_firstlayer_dp as dp


def build_uniform_noise_factors(
    patch_scores: torch.Tensor,
    epsilon: float,
    delta_priv: float,
    delta_mask: float,
    clip_norm: float,
    noise_scale_multiplier: float = 1.0,
) -> tuple[dp.PatchNoiseFactors, dp.AnalyticGaussianCalibration]:
    del delta_mask

    sensitivity = 2.0 * clip_norm
    calibration = dp.calibrate_analytic_matrix_gaussian(
        epsilon=epsilon,
        delta_priv=delta_priv,
        sensitivity=sensitivity,
    )

    base_noise_std = calibration.base_noise_std * float(noise_scale_multiplier)
    covariance_diag = torch.ones_like(patch_scores, dtype=torch.float32)
    row_noise_scales = torch.full_like(covariance_diag, fill_value=base_noise_std, dtype=torch.float32)

    return (
        dp.PatchNoiseFactors(
            covariance_diag=covariance_diag,
            row_noise_scales=row_noise_scales,
            required_min_singular_value=calibration.base_noise_std,
            left_factor_scale=base_noise_std,
        ),
        calibration,
    )


def main() -> None:
    dp.build_patch_noise_factors = build_uniform_noise_factors
    base.dp.build_patch_noise_factors = build_uniform_noise_factors
    base.main()


if __name__ == "__main__":
    main()

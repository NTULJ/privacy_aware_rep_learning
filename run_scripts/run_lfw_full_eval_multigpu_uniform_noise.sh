#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

env \
  EXTRACTOR_SCRIPT="${EXTRACTOR_SCRIPT:-$ROOT_DIR/scripts/extract_qwen_vl_features_uniform_noise.py}" \
  OUTPUT_TAG="${OUTPUT_TAG:-lfw_full_mgpu_uniform_$(date +%Y%m%d_%H%M%S)}" \
  NOISE_SCALE_MULTIPLIER="${NOISE_SCALE_MULTIPLIER:-1.0}" \
  NOISE_DISTRIBUTION="${NOISE_DISTRIBUTION:-uniform_global}" \
  SPATIAL_REWEIGHTING="${SPATIAL_REWEIGHTING:-false}" \
  bash "$ROOT_DIR/run_lfw_full_eval_multigpu.sh"

#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

EPSILON=(1 4 8)
MODEL_ROOT="${PROJECT_ROOT}/after_sft_models"
LOG_ROOT="${PROJECT_ROOT}/eval_log/eval"
pos="vit_hidden_12"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_ROOT}/${RUN_TS}"

for EPS in "${EPSILON[@]}"; do
    MODEL_DIR="${MODEL_ROOT}/eps${EPS}_pos-${pos}_norm1.0_delta1e-5_step915"
    LOG_FILE="${LOG_ROOT}/${RUN_TS}/eps${EPS}_${pos}.log"

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python "${PROJECT_ROOT}/scripts/evaluation/eval_qwen3_vl_models.py" \
        --model_path_override "$MODEL_DIR" \
        --test_fall_path "${PROJECT_ROOT}/data/test.fall.parquet" \
        --test_smoke_path "${PROJECT_ROOT}/data/test.smoke.parquet" \
        --test_fight_path "${PROJECT_ROOT}/data/test.fight.fixed.parquet" \
        --eval_mode structured \
        --prompt_mode dataset \
        --external_lib vision_tower_dp_everywhere \
        --batch_size_falldown 128 \
        --batch_size_smoke 4 \
        --batch_size_fight 32 \
        --num_gpus 8 \
        --dtype bfloat16 \
        > "$LOG_FILE" 2>&1
done

echo "完成，日志: ${LOG_ROOT}/${RUN_TS}/"
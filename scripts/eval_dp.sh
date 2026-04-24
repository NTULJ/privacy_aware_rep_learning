#!/usr/bin/env bash
# =============================================================================
# 统一评测脚本 (替代 eval_eps*.sh / eval_vit_*.sh / eval_after_pos.sh)
# =============================================================================
# 用法:
#   bash scripts/eval_dp.sh --eps 1 --poss vit_input,vit_hidden_8,vit_hidden_16,vit_hidden_24
#   bash scripts/eval_dp.sh --eps 8 --poss vit_hidden_8
#   bash scripts/eval_dp.sh --eps 1,4,8 --poss vit_hidden_4
#   bash scripts/eval_dp.sh --eps 8 --poss vit_hidden_8 --norm-c 11
#   bash scripts/eval_dp.sh --eps 1 --poss vit_hidden_24 --tasks falldown
#   bash scripts/eval_dp.sh --eps 8 --poss vit_hidden_8 --model-root /path/to/models
# =============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
UV_RUN=(uv run)

# 默认参数
EPSILONS=()
POSS=()
NORM_C="1.0"
DELTA="1e-5"
STEP="915"
GPUS="0,1,2,3,4,5,6,7"
NUM_GPUS=8
TASKS=""  # 空=全部任务, 可选: falldown, smoke, fight
MODEL_ROOT=""
SEED=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --eps)     IFS=',' read -ra EPSILONS <<< "$2"; shift 2 ;;
        --poss)    IFS=',' read -ra POSS <<< "$2"; shift 2 ;;
        --norm-c)  NORM_C="$2"; shift 2 ;;
        --delta)   DELTA="$2"; shift 2 ;;
        --step)    STEP="$2"; shift 2 ;;
        --gpus)    GPUS="$2"; shift 2 ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --tasks)   TASKS="$2"; shift 2 ;;
        --model-root) MODEL_ROOT="$2"; shift 2 ;;
        --seed)    SEED="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [ ${#EPSILONS[@]} -eq 0 ]; then
    echo "错误: --eps 是必选参数 (例: --eps 1 或 --eps 1,4,8)"
    exit 1
fi
if [ ${#POSS[@]} -eq 0 ]; then
    echo "错误: --poss 是必选参数 (例: --poss vit_hidden_8 或 --poss vit_input,vit_hidden_8)"
    exit 1
fi

if [ -z "${MODEL_ROOT}" ]; then
    # 默认不再固定到 after_sft_models，直接用项目根目录作为模型根路径。
    MODEL_ROOT="${PROJECT_ROOT}"
fi
LOG_ROOT="${PROJECT_ROOT}/eval_log/eval"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_ROOT}/${RUN_TS}"

# 构建可选的 --tasks 参数
TASK_ARG=""
if [ -n "${TASKS}" ]; then
    TASK_ARG="--tasks ${TASKS}"
fi

for EPS in "${EPSILONS[@]}"; do
    for pos in "${POSS[@]}"; do
        MODEL_DIR="${MODEL_ROOT}/eps${EPS}_pos-${pos}_norm${NORM_C}_delta${DELTA}_step${STEP}"
        LOG_FILE="${LOG_ROOT}/${RUN_TS}/eps${EPS}_${pos}.log"

        echo "=== 评测: eps=${EPS}, pos=${pos} ==="
        echo "模型路径: ${MODEL_DIR}"

        CUDA_VISIBLE_DEVICES=${GPUS} \
        "${UV_RUN[@]}" python "${PROJECT_ROOT}/scripts/evaluation/eval_qwen3_vl_models.py" \
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
            --num_gpus ${NUM_GPUS} \
            --seed "${SEED}" \
            --dtype bfloat16 \
            ${TASK_ARG} \
            > "$LOG_FILE" 2>&1
    done
done

echo "完成，日志: ${LOG_ROOT}/${RUN_TS}/"

#!/usr/bin/env bash
# =============================================================================
# 统一训练编排脚本 (替代 run_eps*.sh / exp_vit_*.sh / exp_after_pos.sh)
# =============================================================================
# 用法:
#   bash scripts/run_exp_batch.sh --eps 1 --poss vit_input,vit_hidden_8,vit_hidden_16,vit_hidden_24
#   bash scripts/run_exp_batch.sh --eps 8 --poss vit_hidden_8
#   bash scripts/run_exp_batch.sh --eps 1,4,8 --poss vit_hidden_4
#   bash scripts/run_exp_batch.sh --eps 8 --poss vit_hidden_8 --use-mlp --mlp-dim 128
# =============================================================================

set -euo pipefail

EPSILONS=()
POSS=()
USE_MLP=false
MLP_DIM=128

while [[ $# -gt 0 ]]; do
    case $1 in
        --eps)     IFS=',' read -ra EPSILONS <<< "$2"; shift 2 ;;
        --poss)    IFS=',' read -ra POSS <<< "$2"; shift 2 ;;
        --use-mlp) USE_MLP=true; shift ;;
        --mlp-dim) MLP_DIM="$2"; shift 2 ;;
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

for eps in "${EPSILONS[@]}"; do
    for pos in "${POSS[@]}"; do
        echo "=== 运行: eps=${eps}, pos=${pos}, mlp=${USE_MLP} ==="
        MLP_ARGS=""
        if [[ "$USE_MLP" == true ]]; then
            MLP_ARGS="--use-mlp --mlp-dim ${MLP_DIM}"
        fi
        if ! bash scripts/exp_pos_eps.sh --type nonfreeze --mode dp --epsilon "${eps}" --pos "${pos}" ${MLP_ARGS}; then
            echo "任务失败，停止"
            exit 1
        fi
    done
done

echo "所有实验完成！"

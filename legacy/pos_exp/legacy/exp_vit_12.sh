#!/usr/bin/env bash
set -euo pipefail

EPSILONS=(1 4 8)
POSS=(vit_hidden_12)  # 可选: vit_after_pos_embed vit_hidden_x vit_output

for eps in "${EPSILONS[@]}"; do
    for pos in "${POSS[@]}"; do
        echo "=== 运行: eps=${eps}, pos=${pos} ==="
        if ! bash scripts/exp_pos_eps.sh --type nonfreeze --mode dp --epsilon "${eps}" --pos "${pos}"; then
            echo "任务失败，停止"
            exit 1
        fi
    done
done

echo "所有实验完成！"
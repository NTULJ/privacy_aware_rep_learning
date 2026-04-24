#!/bin/bash
# 按顺序执行所有实验
EXPENS=(8)
# POSS=("vit_input" "vit_hidden_8" "vit_hidden_16" "vit_hidden_24")  # 可选: vit_after_pos_embed vit_hidden_x vit_output
POSS=("vit_hidden_8")  # 可选: vit_after_pos_embed vit_hidden_x vit_output
for eps in "${EXPENS[@]}"; do
    for pos in "${POSS[@]}"; do
        echo "=== 运行: eps=${eps}, pos=${pos} ==="
        bash scripts/exp_pos_eps_MLP.sh --type nonfreeze --mode dp --epsilon ${eps} --pos ${pos} --mlp-hidden-dim 128
        # 可选：检查是否成功，成功再继续
        if [ $? -ne 0 ]; then
            echo "任务失败，停止"
            exit 1
        fi
    done
done
echo "所有实验完成！"
#!/bin/bash
# 按顺序执行所有实验
EXPENS=(6)
POSS=("vit_input" "vit_hidden_8" "vit_hidden_16" "vit_hidden_24")
for eps in "${EXPENS[@]}"; do
    for pos in "${POSS[@]}"; do
        echo "=== 运行: eps=${eps}, pos=${pos} ==="
        bash scripts/exp_pos_eps.sh --type nonfreeze --mode dp --epsilon ${eps} --pos ${pos}
        wait # 等待当前实验完成再继续下一个
        # 可选：检查是否成功，成功再继续
        if [ $? -ne 0 ]; then
            echo "任务失败，停止"
            exit 1
        fi
    done
done
echo "所有实验完成！"
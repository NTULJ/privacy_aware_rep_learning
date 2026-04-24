#!/bin/bash
set -euo pipefail

# -------------------------------
# 参数解析
# -------------------------------
BASE_DIR=""
TARGET_DIR=""
STEP=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --base_dir) BASE_DIR="$2"; shift 2 ;;
        --target_dir) TARGET_DIR="$2"; shift 2 ;;
        --step) STEP="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [[ -z "$BASE_DIR" || -z "$TARGET_DIR" || -z "$STEP" ]]; then
    echo "错误: 需要提供 --base_dir, --target_dir, --step 参数"
    exit 1
fi

echo "准备合并: BASE_DIR=${BASE_DIR}, STEP=${STEP}, TARGET_DIR=${TARGET_DIR}"

# 合并检查点
python scripts/training/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir "${BASE_DIR}/global_step_${STEP}" \
    --target_dir "${TARGET_DIR}/merged_${STEP}" \
    > "merge_step_${STEP}.log" 2>&1

echo "合并完成: STEP=${STEP}"
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

# ===================== 必需参数 =====================
MODEL_PATH="${MODEL_PATH:-}"
TEST_SMOKE_PATH="${TEST_SMOKE_PATH:-}"
TEST_FALL_PATH="${TEST_FALL_PATH:-}"
TEST_FIGHT_PATH="${TEST_FIGHT_PATH:-}"

# 校验必要参数
if [ -z "${MODEL_PATH}" ]; then
  echo "错误: MODEL_PATH 是必需参数"
  exit 1
fi

# 构建任务列表和参数
TASK_ARGS=()
if [ -n "${TEST_SMOKE_PATH}" ]; then
  TASK_ARGS+=("--test_smoke_path" "${TEST_SMOKE_PATH}")
fi
if [ -n "${TEST_FALL_PATH}" ]; then
  TASK_ARGS+=("--test_fall_path" "${TEST_FALL_PATH}")
fi
if [ -n "${TEST_FIGHT_PATH}" ]; then
  TASK_ARGS+=("--test_fight_path" "${TEST_FIGHT_PATH}")
fi

# 如果没有指定任何测试集，使用默认值
if [ ${#TASK_ARGS[@]} -eq 0 ]; then
  echo "注意: 未指定测试集路径，将使用默认路径"
fi

# ===================== GPU 配置 =====================
GPUS_CSV="${GPUS_CSV:-0}"
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"

# ===================== 日志目录 =====================
RUN_TS="$(date +%Y%m%d_%H%M%S)"
MODEL_NAME="$(basename "${MODEL_PATH}")"
LOG_DIR="${SCRIPT_DIR}/../eval_log/eval/${RUN_TS}/${MODEL_NAME}"
mkdir -p "${LOG_DIR}"

echo "=== 并行评测启动 ==="
echo "MODEL_PATH: ${MODEL_PATH}"
echo "TEST_SMOKE_PATH: ${TEST_SMOKE_PATH:-未指定}"
echo "TEST_FALL_PATH: ${TEST_FALL_PATH:-未指定}"
echo "TEST_FIGHT_PATH: ${TEST_FIGHT_PATH:-未指定}"
echo "LOG_DIR: ${LOG_DIR}"

# ===================== 启动评测 =====================
echo "[启动] model=${MODEL_NAME}, gpu=${GPUS[0]}"
log_file="${LOG_DIR}/eval.log"

CUDA_VISIBLE_DEVICES="${GPUS[0]}" python "${SCRIPT_DIR}/evaluation/eval_qwen3_vl_models.py" \
  --eval_mode structured \
  --prompt_mode dataset \
  --model_path_override "${MODEL_PATH}" \
  --device cuda \
  --dtype bfloat16 \
  "${TASK_ARGS[@]}" \
  > "${log_file}" 2>&1

echo "评测完成。日志: ${log_file}"

# 打印日志内容（关键结果）
echo ""
echo "=== 评测结果 ==="
if [ -f "${log_file}" ]; then
  grep -E "^\[|total=|accuracy=|f1=" "${log_file}" | head -20
fi
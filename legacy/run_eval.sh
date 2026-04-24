# 修正变量定义和引用
fold_dir="20260415_135454"
BASE_PATH="outputs/sft/qwen3-vl-baseline/${fold_dir}/merged/nonfreeze"

# --- 并行组 (GPU 2-7) ---
# 建议加上 & 符号后的等待，或者明确它们是独立运行的
MODEL_PATH=${BASE_PATH}/merged_step_baseline_step549 GPUS_CSV=2 bash scripts/test.sh &
MODEL_PATH=${BASE_PATH}/merged_step_baseline_step732 GPUS_CSV=3 bash scripts/test.sh &
MODEL_PATH=${BASE_PATH}/merged_step_baseline_step915 GPUS_CSV=4 bash scripts/test.sh &
MODEL_PATH=${BASE_PATH}/merged_step_baseline_step1098 GPUS_CSV=5 bash scripts/test.sh &
MODEL_PATH=${BASE_PATH}/merged_step_baseline_step1281 GPUS_CSV=6 bash scripts/test.sh &
MODEL_PATH=${BASE_PATH}/merged_step_baseline_step1464 GPUS_CSV=7 bash scripts/test.sh &

# --- 串行组1 (GPU 0) ---
{
    MODEL_PATH=${BASE_PATH}/merged_step_baseline_step183 GPUS_CSV=0 bash scripts/test.sh
    MODEL_PATH=${BASE_PATH}/merged_step_baseline_step1647 GPUS_CSV=0 bash scripts/test.sh
} &

# --- 串行组2 (GPU 1) ---
{
    MODEL_PATH=${BASE_PATH}/merged_step_baseline_step366 GPUS_CSV=1 bash scripts/test.sh
    MODEL_PATH=${BASE_PATH}/merged_step_baseline_step1830 GPUS_CSV=1 bash scripts/test.sh
} &

# 等待所有后台任务完成
wait
echo "所有测试任务已完成。"
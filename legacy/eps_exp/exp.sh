#!/usr/bin/env bash
# =============================================================================
# epsilon 实验统一训练脚本
# =============================================================================
# 用法示例:
#   ./run_exp.sh --type freeze --epsilon 1
#   ./run_exp.sh --type nonfreeze --epsilon 8
#   ./run_exp.sh --type nonfreeze --mode baseline
#   ./run_exp.sh --type freeze --epsilon inf
#
# 参数说明:
#   --type      [必选] freeze 或 nonfreeze
#   --epsilon   [可选] epsilon 值 (0, 1, 2, 4, 8, 40, inf) (默认: 1)
#   --mode      [可选] dp 或 baseline (默认: dp)
#   --delta     [可选] delta 值 (默认: 1e-5)
# =============================================================================

set -xeuo pipefail

# -------------------------------
# 1. 解析参数
# -------------------------------
TYPE=""
EPSILON=1
MODE="dp"
DELTA="1e-5"
LOCAL_DP=true
NOISE_TYPE=aGM
NORM_C=1.0
MODEL_ID="${MODEL_ID:-/gemini/space/Models/Qwen/Qwen3-VL-8B-Instruct}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --type) TYPE="$2"; shift 2 ;;
        --epsilon) EPSILON="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --delta) DELTA="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# 验证必选参数
if [[ -z "$TYPE" ]]; then
    echo "错误: --type 是必选参数 (freeze 或 nonfreeze)"
    exit 1
fi

if [[ "$TYPE" != "freeze" && "$TYPE" != "nonfreeze" ]]; then
    echo "错误: --type 必须是 freeze 或 nonfreeze"
    exit 1
fi

# -------------------------------
# 2. 公共配置
# -------------------------------
export http_proxy=http://192.168.48.123:3128/
export https_proxy=http://192.168.48.123:3128/

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export WANDB_API_KEY=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
export WANDB_DIR="${PROJECT_ROOT}/wandb"

# 时间戳
RUN_TS="$(date +%Y%m%d_%H%M%S)"

# -------------------------------
# 3. 根据 type 设置 ENTRYPOINT 和 MERGE_BASE_DIR
# -------------------------------
if [[ "$TYPE" == "freeze" ]]; then
    ENTRYPOINT="-m verl.trainer.sft_trainer_freeze"
    MERGE_SUFFIX="freeze"
else
    ENTRYPOINT="-m verl.trainer.sft_trainer"
    MERGE_SUFFIX="nonfreeze"
fi

# -------------------------------
# 4. 根据 mode 设置 project_name
# -------------------------------
if [[ "$MODE" == "baseline" ]]; then
    RUN_NAME="baseline"
    EXP_NAME="baseline"
    PROJECT_NAME="qwen3-vl-baseline"
else
    if [[ "$EPSILON" == "inf" ]]; then
        EPSILON_DISPLAY=700
        EXP_NAME="epsinf"
    else
        EPSILON_DISPLAY=$EPSILON
        EXP_NAME="eps${EPSILON}"
    fi
    PROJECT_NAME="qwen3-vl-dp"
fi

MERGE_BASE_DIR="${PROJECT_ROOT}/outputs/sft/${PROJECT_NAME}/${RUN_TS}/merged/${MERGE_SUFFIX}"

# 日志目录
LOG_DIR="${PROJECT_ROOT}/eval_log/eval/${RUN_TS}/${TYPE}_${RUN_NAME:-eps${EPSILON}}"
EVAL_SUFFIX="_${TYPE}_${RUN_NAME:-eps${EPSILON}}_${RUN_TS}"
mkdir -p "${LOG_DIR}"

echo "============================================"
echo "epsilon 实验配置"
echo "============================================"
echo "TYPE:        $TYPE"
echo "EPSILON:     $EPSILON"
echo "MODE:        $MODE"
echo "DELTA:       $DELTA"
echo "LOCAL_DP:    $LOCAL_DP"
echo "NOISE_TYPE:  $NOISE_TYPE"
echo "NORM_C:      $NORM_C"
echo "ENTRYPOINT:  $ENTRYPOINT"
echo "MERGE_BASE:  $MERGE_BASE_DIR"
echo "LOG_DIR:     $LOG_DIR"

# 统计训练集数量
echo ""
echo "训练数据集样本数:"
FALL_COUNT=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('${PROJECT_ROOT}/data/train.fall.parquet')))")
FIGHT_COUNT=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('${PROJECT_ROOT}/data/train.fight.fixed.parquet')))")
SMOKE_COUNT=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('${PROJECT_ROOT}/data/train.smoke.parquet')))")
echo "  train.fall.parquet:  ${FALL_COUNT} 条"
echo "  train.fight.fixed.parquet: ${FIGHT_COUNT} 条"
echo "  train.smoke.parquet: ${SMOKE_COUNT} 条"
echo "  总计: $((FALL_COUNT + FIGHT_COUNT + SMOKE_COUNT)) 条"
echo "============================================"

# -------------------------------
# 5. 训练数据
# -------------------------------
TRAIN_FILES="${PROJECT_ROOT}/data/train.fall.parquet,${PROJECT_ROOT}/data/train.fight.fixed.parquet,${PROJECT_ROOT}/data/train.smoke.parquet"
FILES_JOINED="[${TRAIN_FILES}]"

# -------------------------------
# 6. 引擎配置
# -------------------------------
BACKEND=${BACKEND:-fsdp}

COMMON_OPTIM_ARGS="
    optim.lr=2e-5 \
    optim.lr_warmup_steps_ratio=0.01 \
    optim.weight_decay=0.1 \
    optim.betas=[0.9,0.95] \
    optim.clip_grad=1.0 \
    optim.min_lr_ratio=0.1 \
    optim.warmup_style=cosine"

if [[ "$BACKEND" == "fsdp" ]]; then
    echo "--- Using FSDP Engine ---"
    ENGINE_CONFIG="
        engine=fsdp \
        optim=fsdp \
        ${COMMON_OPTIM_ARGS} \
        engine.ulysses_sequence_parallel_size=${SP_SIZE:-1} \
        engine.strategy=${FSDP_STRATEGY:-fsdp2} \
        engine.fsdp_size=${FSDP_SIZE:--1} \
        engine.model_dtype=bfloat16 \
        engine.dtype=bfloat16"
else
    echo "Megatron engine is not fully configured in this script yet."
    exit -1
fi

# -------------------------------
# 7. 检查点路径
# -------------------------------
MODEL_ID="${MODEL_ID:-/gemini/space/Models/Qwen/Qwen3-VL-8B-Instruct}"
CKPT_HOME="${PROJECT_ROOT}/outputs/sft/${PROJECT_NAME}/${RUN_TS}/${MERGE_SUFFIX}/${EXP_NAME}"
mkdir -p "${CKPT_HOME}"

# -------------------------------
# 8. 根据 mode 添加 model 参数
# -------------------------------
if [[ "$MODE" == "baseline" ]]; then
    MODEL_PARAMS="model.path=$MODEL_ID"
else
    MODEL_PARAMS="model.path=$MODEL_ID \
model.external_lib=vision_tower_dp \
+model.override_config.vision_dp_enable=true \
+model.override_config.vision_dp_epsilon=${EPSILON} \
+model.override_config.vision_dp_delta=${DELTA} \
+model.override_config.vision_dp_local_dp=${LOCAL_DP} \
+model.override_config.vision_dp_noise_type=${NOISE_TYPE} \
+model.override_config.vision_dp_norm_c=${NORM_C}"
fi

# -------------------------------
# 8. 执行训练
# -------------------------------
torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_TRAINERS:-8} \
    ${ENTRYPOINT} \
    data.train_files="${FILES_JOINED}" \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=4 \
    data.max_length=2048 \
    data.use_dynamic_bsz=False \
    ${ENGINE_CONFIG} \
    ${MODEL_PARAMS} \
    trainer.test_freq=-1 \
    trainer.save_freq="after_each_epoch" \
    trainer.logger=[console,wandb] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.total_epochs=10 \
    trainer.default_local_dir="${CKPT_HOME}" \
    checkpoint.save_contents=[model,optimizer,extra] \
    "$@"

echo "训练完成! 检查点保存于: ${CKPT_HOME}"

# -------------------------------
# 9. 合并检查点
# -------------------------------
echo ""
echo ">>> 开始并行合并 Checkpoints..."

# 获取所有 global_step 目录
if [ -d "$CKPT_HOME" ]; then
    echo "正在从 $CKPT_HOME 自动提取 Steps..."
    mapfile -t STEPS < <(ls -d "${CKPT_HOME}"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n)
else
    echo "错误：训练目录 $CKPT_HOME 不存在，无法搜集 Steps。"
    exit 1
fi

if [ ${#STEPS[@]} -eq 0 ]; then
    echo "警告：在 $CKPT_HOME 下未发现任何 global_step_* 目录！"
    exit 1
fi

echo "探测到以下 Steps: ${STEPS[*]}"

is_merged_model_ready() {
    local model_dir="$1"
    if [ ! -d "$model_dir" ]; then
        return 1
    fi
    if [ -f "${model_dir}/config.json" ] && [ -f "${model_dir}/tokenizer_config.json" ] && \
       { [ -f "${model_dir}/model.safetensors.index.json" ] || [ -f "${model_dir}/model.safetensors" ] || compgen -G "${model_dir}/model-*.safetensors" > /dev/null; }; then
        return 0
    fi
    return 1
}

MAX_JOBS=2
for i in "${!STEPS[@]}"; do
    STEP=${STEPS[$i]}
    (
        if [[ "$MODE" == "baseline" ]]; then
            TARGET_DIR="${MERGE_BASE_DIR}/merged_step_${EXP_NAME}_step${STEP}"
        else
            TARGET_DIR="${MERGE_BASE_DIR}/merged_step_epsilon${EPSILON}_step${STEP}"
        fi

        if is_merged_model_ready "${TARGET_DIR}"; then
            echo "-> [Job $i] Step ${STEP} 已存在可用 merged 模型，跳过合并: ${TARGET_DIR}"
            exit 0
        fi

        echo "-> [Job $i] 正在合并 Step: ${STEP} 到 ${TARGET_DIR}"
        mkdir -p "${TARGET_DIR}"
        
        python "${PROJECT_ROOT}/scripts/training/legacy_model_merger.py" merge \
            --backend fsdp \
            --local_dir "${CKPT_HOME}/global_step_${STEP}" \
            --target_dir "${TARGET_DIR}" \
            > "${LOG_DIR}/merge_step_${STEP}.log" 2>&1
    ) &

    if (( (i + 1) % MAX_JOBS == 0 )); then
        wait
    fi
done
wait

echo "所有合并任务完成！"
echo "合并后模型保存于: ${MERGE_BASE_DIR}"
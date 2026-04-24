#!/usr/bin/env bash
# =============================================================================
# epsilon + position 实验统一训练脚本
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
MODEL_ID="${MODEL_ID:-/gemini/space/Models/Qwen/Qwen3-VL-8B-Instruct}"
pos="vit_output"
MLP_HIDDEN_DIM=128
NORM_C=11
while [[ $# -gt 0 ]]; do
    case $1 in
        --type) TYPE="$2"; shift 2 ;;
        --epsilon) EPSILON="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --delta) DELTA="$2"; shift 2 ;;
        --pos) pos="$2"; shift 2 ;;
        --mlp-hidden-dim) MLP_HIDDEN_DIM="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [[ -z "$TYPE" ]] || [[ "$TYPE" != "freeze" && "$TYPE" != "nonfreeze" ]]; then
    echo "错误: --type 必须是 freeze 或 nonfreeze"
    exit 1
fi

# -------------------------------
# 2. 公共配置 & 环境变量
# -------------------------------
export http_proxy=http://192.168.48.123:3128/
export https_proxy=http://192.168.48.123:3128/

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export WANDB_API_KEY=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
export WANDB_DIR="${PROJECT_ROOT}/wandb"

RUN_TS="$(date +%Y%m%d_%H%M%S)"

# -------------------------------
# 3. 生成唯一的 EXP_ID (核心修改点)
# -------------------------------
if [[ "$TYPE" == "freeze" ]]; then
    ENTRYPOINT="-m verl.trainer.sft_trainer_freeze"
else
    ENTRYPOINT="-m verl.trainer.sft_trainer"
fi

if [[ "$MODE" == "baseline" ]]; then
    PROJECT_NAME="qwen3-vl-baseline"
    EXP_ID="baseline"
else
    PROJECT_NAME="qwen3-vl-dp"
    if [[ "$EPSILON" == "inf" ]]; then
        EXP_ID="epsinf_pos-${pos}"
    else
        EXP_ID="eps${EPSILON}_pos-${pos}_norm${NORM_C}_delta${DELTA}"
    fi
fi

# 统一规划所有路径，全部带上 EXP_ID 保证绝对不冲突
MERGE_BASE_DIR="${PROJECT_ROOT}/outputs/sft/${PROJECT_NAME}/${RUN_TS}/merged/${TYPE}"
CKPT_HOME="${PROJECT_ROOT}/outputs/sft/${PROJECT_NAME}/${RUN_TS}/${TYPE}/${EXP_ID}"
TRAIN_LOG_DIR="${PROJECT_ROOT}/train_log/${RUN_TS}"
LOG_FILE="${TRAIN_LOG_DIR}/${TYPE}_${EXP_ID}.log"
MERGE_LOG_DIR="${PROJECT_ROOT}/eval_log/eval/${RUN_TS}/${TYPE}_${EXP_ID}"

mkdir -p "${CKPT_HOME}" "${TRAIN_LOG_DIR}" "${MERGE_LOG_DIR}"

echo "============================================"
echo "当前实验唯一标识 (EXP_ID): $EXP_ID"
echo "检查点路径: $CKPT_HOME"
echo "训练日志: $LOG_FILE"
echo "合并模型根目录: $MERGE_BASE_DIR"
echo "============================================"

# -------------------------------
# 4. 引擎与模型配置
# -------------------------------
TRAIN_FILES="${PROJECT_ROOT}/data/train.fall.parquet,${PROJECT_ROOT}/data/train.fight.fixed.parquet,${PROJECT_ROOT}/data/train.smoke.parquet"
FILES_JOINED="[${TRAIN_FILES}]"
BACKEND=${BACKEND:-fsdp}

if [[ "$BACKEND" == "fsdp" ]]; then
    ENGINE_CONFIG="engine=fsdp optim=fsdp optim.lr=2e-5 optim.lr_warmup_steps_ratio=0.01 optim.weight_decay=0.1 optim.betas=[0.9,0.95] optim.clip_grad=1.0 optim.min_lr_ratio=0.1 optim.warmup_style=cosine engine.ulysses_sequence_parallel_size=${SP_SIZE:-1} engine.strategy=${FSDP_STRATEGY:-fsdp2} engine.fsdp_size=${FSDP_SIZE:--1} engine.model_dtype=bfloat16 engine.dtype=bfloat16  engine.use_orig_params=True"
else
    echo "只支持 fsdp"; exit -1
fi

if [[ "$MODE" == "baseline" ]]; then
    MODEL_PARAMS="model.path=$MODEL_ID"
else
    MODEL_PARAMS="model.path=$MODEL_ID \
    model.external_lib=vision_tower_dp_everywhere \
    +model.override_config.vision_dp_enable=true \
    +model.override_config.vision_dp_epsilon=${EPSILON} \
    +model.override_config.vision_dp_delta=${DELTA} \
    +model.override_config.vision_dp_local_dp=${LOCAL_DP} \
    +model.override_config.vision_dp_noise_type=${NOISE_TYPE} \
    +model.override_config.vision_dp_norm_c=${NORM_C} \
    +model.override_config.vision_dp_noise_location=${pos} \
    +model.override_config.vision_dp_use_mlp_subspace=true \
    +model.override_config.vision_dp_mlp_hidden_dim=${MLP_HIDDEN_DIM}"
fi

# -------------------------------
# 5. 执行训练
# -------------------------------
echo "启动训练..."
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
    trainer.experiment_name="${EXP_ID}" \
    trainer.total_epochs=5 \
    trainer.default_local_dir="${CKPT_HOME}" \
    checkpoint.save_contents=[model,optimizer,extra] \
    "$@" > "${LOG_FILE}" 2>&1 &

PID=$!
echo "训练进程 PID: ${PID}。正在后台运行，等待完成..."
wait $PID
echo "训练环节结束。"

# -------------------------------
# 6. 合并检查点
# -------------------------------
echo ">>> 开始自动合并 Checkpoints..."

if [ ! -d "$CKPT_HOME" ]; then
    echo "未发现训练输出目录，跳过合并。"
    exit 1
fi

mapfile -t STEPS < <(ls -d "${CKPT_HOME}"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n)

if [ ${#STEPS[@]} -eq 0 ]; then
    echo "警告：在 ${CKPT_HOME} 下未找到 global_step 目录，请检查模型是否以 epoch 格式保存。"
    exit 1
fi

MAX_JOBS=2
for i in "${!STEPS[@]}"; do
    STEP=${STEPS[$i]}
    (
        # 合并后的名字直接采用 EXP_ID，彻底杜绝冲突
        TARGET_DIR="${MERGE_BASE_DIR}/${EXP_ID}_step${STEP}"

        if [ -f "${TARGET_DIR}/config.json" ]; then
            echo "-> Step ${STEP} 已经合并过了，跳过。"
            exit 0
        fi

        echo "-> 正在合并 Step: ${STEP} 到 ${TARGET_DIR}"
        mkdir -p "${TARGET_DIR}"
        
        python "${PROJECT_ROOT}/scripts/training/legacy_model_merger.py" merge \
            --backend fsdp \
            --local_dir "${CKPT_HOME}/global_step_${STEP}" \
            --target_dir "${TARGET_DIR}" \
            > "${MERGE_LOG_DIR}/merge_step_${STEP}.log" 2>&1
    ) &

    if (( (i + 1) % MAX_JOBS == 0 )); then
        wait
    fi
done
wait

echo "本次实验 (${EXP_ID}) 全流程结束！"
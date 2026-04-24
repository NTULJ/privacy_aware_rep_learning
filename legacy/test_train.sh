#!/usr/bin/env bash
# =============================================================================
# 快速训练测试脚本 — train.fall.parquet + fixed PCA
# =============================================================================
# 用法:
#   bash scripts/test_train.sh --pos vit_hidden_8 --epsilon 1
#   bash scripts/test_train.sh --pos vit_output  --epsilon 8 --pca-basis-dir /tmp/basis_multi
# =============================================================================

set -xeuo pipefail

# -------------------------------
# 1. 参数
# -------------------------------
TYPE="nonfreeze"
EPSILON=1
MODE="dp"
DELTA="1e-5"
LOCAL_DP=true
NOISE_TYPE=aGM
NORM_C=1.0
USE_MLP=false
MLP_HIDDEN_DIM=128
USE_PCA=false
PCA_BASIS_DIR=""
PCA_RANK=128
PCA_SAMPLES=256
pos="vit_hidden_8"
EPOCHS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --type)        TYPE="$2"; shift 2 ;;
        --epsilon)     EPSILON="$2"; shift 2 ;;
        --pos)         pos="$2"; shift 2 ;;
        --norm-c)      NORM_C="$2"; shift 2 ;;
        --use-mlp)     USE_MLP=true; shift ;;
        --mlp-dim)     MLP_HIDDEN_DIM="$2"; shift 2 ;;
        --use-pca)     USE_PCA=true; shift ;;
        --pca-basis-dir) PCA_BASIS_DIR="$2"; shift 2 ;;
        --pca-rank)    PCA_RANK="$2"; shift 2 ;;
        --pca-samples) PCA_SAMPLES="$2"; shift 2 ;;
        --epochs)      EPOCHS="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# -------------------------------
# 2. 环境
# -------------------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export WANDB_API_KEY=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
export WANDB_DIR="${PROJECT_ROOT}/wandb"

MODEL_ID="${MODEL_ID:-/workspace/s/ddn/gemini/gemini-sharedata/space/wqmu4k88unnm/Models/Qwen3-VL-8B-Instruct}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"

# -------------------------------
# 3. 入口 & EXP_ID
# -------------------------------
if [[ "$USE_MLP" == true && "$USE_PCA" == true ]]; then
    echo "错误: --use-mlp 和 --use-pca 不能同时启用，二者互斥"
    exit 1
fi

if [[ "$TYPE" == "freeze" ]]; then
    ENTRYPOINT="-m verl.trainer.sft_trainer_freeze"
else
    ENTRYPOINT="-m verl.trainer.sft_trainer"
fi

PROJECT_NAME="qwen3-vl-dp"
EXP_ID="eps${EPSILON}_pos-${pos}_norm${NORM_C}"
if [[ "$USE_MLP" == true ]]; then
    EXP_ID="${EXP_ID}_mlp${MLP_HIDDEN_DIM}"
fi
if [[ "$USE_PCA" == true ]]; then
    EXP_ID="${EXP_ID}_pca"
fi

CKPT_HOME="${PROJECT_ROOT}/outputs/sft/${PROJECT_NAME}/${RUN_TS}/${TYPE}/${EXP_ID}"
TRAIN_LOG_DIR="${PROJECT_ROOT}/train_log/${RUN_TS}"
LOG_FILE="${TRAIN_LOG_DIR}/${TYPE}_${EXP_ID}.log"
MERGE_BASE_DIR="${PROJECT_ROOT}/outputs/sft/${PROJECT_NAME}/${RUN_TS}/merged/${TYPE}"
MERGE_LOG_DIR="${PROJECT_ROOT}/eval_log/eval/${RUN_TS}/${TYPE}_${EXP_ID}"

mkdir -p "${CKPT_HOME}" "${TRAIN_LOG_DIR}" "${MERGE_LOG_DIR}"

echo "============================================"
echo "EXP_ID: $EXP_ID"
echo "位置: $pos  epsilon: $EPSILON"
echo "MLP: $USE_MLP (dim=${MLP_HIDDEN_DIM})"
echo "PCA: $USE_PCA (basis_dir=${PCA_BASIS_DIR:-auto})"
echo "检查点: $CKPT_HOME"
echo "日志: $LOG_FILE"
echo "============================================"

# -------------------------------
# 4. Fit PCA basis (if needed)
# -------------------------------
if [[ "$USE_PCA" == true && -z "$PCA_BASIS_DIR" ]]; then
    PCA_BASIS_DIR="${PROJECT_ROOT}/outputs/pca_basis/${RUN_TS}_${EXP_ID}"
    echo ">>> 未指定 --pca-basis-dir，自动拟合 PCA basis -> ${PCA_BASIS_DIR}"
    python "${PROJECT_ROOT}/scripts/fit_pca_basis.py" \
        --model "$MODEL_ID" \
        --locations "$pos" \
        --rank "$PCA_RANK" \
        --samples "$PCA_SAMPLES" \
        --out "$PCA_BASIS_DIR"
    echo ">>> PCA basis 拟合完成"
fi

# -------------------------------
# 5. 模型配置
# -------------------------------
TRAIN_FILES="${PROJECT_ROOT}/data/train.fall.parquet"
FILES_JOINED="[${TRAIN_FILES}]"
BACKEND=${BACKEND:-fsdp}

ENGINE_CONFIG="engine=fsdp optim=fsdp optim.lr=2e-5 optim.lr_warmup_steps_ratio=0.01 optim.weight_decay=0.1 optim.betas=[0.9,0.95] optim.clip_grad=1.0 optim.min_lr_ratio=0.1 optim.warmup_style=cosine engine.ulysses_sequence_parallel_size=${SP_SIZE:-1} engine.strategy=${FSDP_STRATEGY:-fsdp2} engine.fsdp_size=${FSDP_SIZE:--1} engine.model_dtype=bfloat16 engine.dtype=bfloat16 engine.use_orig_params=True"

MODEL_PARAMS="model.path=$MODEL_ID \
model.external_lib=vision_tower_dp_everywhere \
+model.override_config.vision_dp_enable=true \
+model.override_config.vision_dp_epsilon=${EPSILON} \
+model.override_config.vision_dp_delta=${DELTA} \
+model.override_config.vision_dp_local_dp=${LOCAL_DP} \
+model.override_config.vision_dp_noise_type=${NOISE_TYPE} \
+model.override_config.vision_dp_norm_c=${NORM_C} \
+model.override_config.vision_dp_noise_location=${pos}"

if [[ "$USE_MLP" == true ]]; then
    MODEL_PARAMS="${MODEL_PARAMS} \
+model.override_config.vision_dp_use_mlp_subspace=true \
+model.override_config.vision_dp_mlp_hidden_dim=${MLP_HIDDEN_DIM}"
elif [[ "$USE_PCA" == true ]]; then
    MODEL_PARAMS="${MODEL_PARAMS} \
+model.override_config.vision_dp_use_fixed_pca=true \
+model.override_config.vision_dp_pca_basis_dir=${PCA_BASIS_DIR}"
fi

# -------------------------------
# 6. 训练
# -------------------------------
SAVE_FREQ="after_each_epoch"
if [[ "$USE_MLP" == true ]]; then
    SAVE_FREQ=1
fi

echo ">>> 启动训练..."
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
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.logger=[console,wandb] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_ID}" \
    trainer.total_epochs=${EPOCHS} \
    trainer.default_local_dir="${CKPT_HOME}" \
    checkpoint.save_contents=[model,optimizer,extra] \
    "$@" > "${LOG_FILE}" 2>&1 &

PID=$!
echo "训练进程 PID: ${PID}"
wait $PID
echo ">>> 训练结束"

# -------------------------------
# 7. 合并
# -------------------------------
echo ">>> 开始合并 Checkpoints..."

mapfile -t STEPS < <(ls -d "${CKPT_HOME}"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n)

if [ ${#STEPS[@]} -eq 0 ]; then
    echo "未找到 global_step 目录，跳过合并。"
    exit 0
fi

MAX_JOBS=2
for i in "${!STEPS[@]}"; do
    STEP=${STEPS[$i]}
    (
        TARGET_DIR="${MERGE_BASE_DIR}/${EXP_ID}_step${STEP}"

        if [ -f "${TARGET_DIR}/config.json" ]; then
            echo "-> Step ${STEP} 已合并，跳过。"
            exit 0
        fi

        echo "-> 合并 Step: ${STEP} -> ${TARGET_DIR}"
        mkdir -p "${TARGET_DIR}"

        python "${PROJECT_ROOT}/scripts/training/legacy_model_merger.py" merge \
            --backend fsdp \
            --local_dir "${CKPT_HOME}/global_step_${STEP}" \
            --target_dir "${TARGET_DIR}" \
            > "${MERGE_LOG_DIR}/merge_step_${STEP}.log" 2>&1

        # 如果用了 PCA，把 basis 复制进合并后的模型目录
        if [[ "$USE_PCA" == true && -n "$PCA_BASIS_DIR" && -d "$PCA_BASIS_DIR" ]]; then
            cp -r "${PCA_BASIS_DIR}" "${TARGET_DIR}/dp_pca_basis"
            echo "-> 已复制 PCA basis 到 ${TARGET_DIR}/dp_pca_basis/"
        fi
    ) &

    if (( (i + 1) % MAX_JOBS == 0 )); then
        wait
    fi
done
wait

echo ">>> 全流程结束！"

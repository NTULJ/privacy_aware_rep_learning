#!/usr/bin/env bash
# =============================================================================
# epsilon + position 实验统一训练脚本
# =============================================================================
# 用法:
#   bash scripts/exp_pos_eps.sh --type nonfreeze --mode dp --epsilon 1 --pos vit_hidden_8
#   bash scripts/exp_pos_eps.sh --type nonfreeze --mode dp --epsilon 8 --pos vit_hidden_8 --use-mlp --mlp-dim 128
#   bash scripts/exp_pos_eps.sh --type nonfreeze --mode dp --epsilon 8 --pos vit_hidden_8 --use-pca --pca-basis-dir /path/to/pca_basis
#   bash scripts/exp_pos_eps.sh --type nonfreeze --mode dp --epsilon 8 --pos vit_hidden_8 --use-pca
#   bash scripts/exp_pos_eps.sh --type nonfreeze --mode dp --epsilon 8 --pos vit_hidden_8 --skip-eval
# =============================================================================

set -xeuo pipefail

# -------------------------------
# 1. 解析参数
# -------------------------------
TYPE="nonfreeze"
EPSILON=8
MODE="dp"
DELTA="1e-5"
LOCAL_DP=true
NOISE_TYPE=aGM
NORM_C=8.0
USE_MLP=false
MLP_HIDDEN_DIM=128
USE_PCA=false
PCA_BASIS_DIR=""
PCA_RANK=128
PCA_SAMPLES=256
SEED=1
PCA_IMAGES_DIR=""
RUN_EVAL=true
EVAL_GPUS="0,1,2,3,4,5,6,7"
EVAL_NUM_GPUS=8
EVAL_TASKS=""
EVAL_BATCH_FALLDOWN=128
EVAL_BATCH_SMOKE=4
EVAL_BATCH_FIGHT=32
MODEL_ID="${MODEL_ID:-/workspace/s/ddn/gemini/gemini-sharedata/space/wqmu4k88unnm/Models/Qwen3-VL-8B-Instruct}"
pos="vit_hidden_8"

while [[ $# -gt 0 ]]; do
    case $1 in
        --type)      TYPE="$2"; shift 2 ;;
        --epsilon)   EPSILON="$2"; shift 2 ;;
        --mode)      MODE="$2"; shift 2 ;;
        --delta)     DELTA="$2"; shift 2 ;;
        --pos)       pos="$2"; shift 2 ;;
        --norm-c)    NORM_C="$2"; shift 2 ;;
        --use-mlp)   USE_MLP=true; shift ;;
        --mlp-dim)   MLP_HIDDEN_DIM="$2"; shift 2 ;;
        --use-pca)   USE_PCA=true; shift ;;
        --pca-basis-dir) PCA_BASIS_DIR="$2"; shift 2 ;;
        --pca-rank)  PCA_RANK="$2"; shift 2 ;;
        --pca-samples) PCA_SAMPLES="$2"; shift 2 ;;
        --seed)      SEED="$2"; shift 2 ;;
        --pca-images-dir) PCA_IMAGES_DIR="$2"; shift 2 ;;
        --skip-eval) RUN_EVAL=false; shift ;;
        --eval-gpus) EVAL_GPUS="$2"; shift 2 ;;
        --eval-num-gpus) EVAL_NUM_GPUS="$2"; shift 2 ;;
        --eval-tasks) EVAL_TASKS="$2"; shift 2 ;;
        --eval-batch-falldown) EVAL_BATCH_FALLDOWN="$2"; shift 2 ;;
        --eval-batch-smoke) EVAL_BATCH_SMOKE="$2"; shift 2 ;;
        --eval-batch-fight) EVAL_BATCH_FIGHT="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [[ -z "$TYPE" ]] || [[ "$TYPE" != "freeze" && "$TYPE" != "nonfreeze" ]]; then
    echo "错误: --type 必须是 freeze 或 nonfreeze"
    exit 1
fi

if [[ "$USE_MLP" == true && "$USE_PCA" == true ]]; then
    echo "错误: --use-mlp 和 --use-pca 互斥，不能同时启用"
    exit 1
fi

# -------------------------------
# 2. 公共配置 & 环境变量
# -------------------------------

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export WANDB_API_KEY=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
export WANDB_DIR="${PROJECT_ROOT}/wandb"
UV_RUN=(uv run)

RUN_TS="$(date +%Y%m%d_%H%M%S)"

# -------------------------------
# 3. 生成唯一的 EXP_ID
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
    if [[ "$USE_MLP" == true ]]; then
        EXP_ID="${EXP_ID}_mlp${MLP_HIDDEN_DIM}"
    elif [[ "$USE_PCA" == true ]]; then
        EXP_ID="${EXP_ID}_pca"
    fi
fi

# 统一规划所有路径，全部带上 EXP_ID 保证绝对不冲突
MERGE_BASE_DIR="${PROJECT_ROOT}/outputs/sft/${PROJECT_NAME}/${RUN_TS}/merged/${TYPE}"
CKPT_HOME="${PROJECT_ROOT}/outputs/sft/${PROJECT_NAME}/${RUN_TS}/${TYPE}/${EXP_ID}"
LOG_ROOT="${PROJECT_ROOT}/log"
TRAIN_LOG_DIR="${LOG_ROOT}/train_log/${RUN_TS}"
EVAL_LOG_DIR="${LOG_ROOT}/eval_log/${RUN_TS}/${TYPE}_${EXP_ID}"
MERGE_LOG_DIR="${LOG_ROOT}/merge_log/${RUN_TS}/${TYPE}_${EXP_ID}"
LOG_FILE="${TRAIN_LOG_DIR}/${TYPE}_${EXP_ID}.log"

if [[ "$USE_PCA" == true && -z "$PCA_BASIS_DIR" ]]; then
    # 默认放在与当次训练输出/merge 同一个 RUN_TS 下，便于归档和复现实验。
    PCA_BASIS_DIR="${PROJECT_ROOT}/outputs/sft/${PROJECT_NAME}/${RUN_TS}/pca_basis/${EXP_ID}"
fi

mkdir -p "${CKPT_HOME}" "${TRAIN_LOG_DIR}" "${EVAL_LOG_DIR}" "${MERGE_LOG_DIR}"

# 先创建训练日志并写入头部；后续 PCA 拟合与 torchrun 一律追加 (>>)，避免拟合输出被训练阶段截断。
{
    echo "============================================"
    echo "RUN START $(date -Is)"
    echo "EXP_ID=${EXP_ID} TYPE=${TYPE} MODE=${MODE}"
    echo "MODEL_ID=${MODEL_ID}"
    echo "pos=${pos} epsilon=${EPSILON} norm_c=${NORM_C} delta=${DELTA}"
    echo "USE_MLP=${USE_MLP} MLP_HIDDEN_DIM=${MLP_HIDDEN_DIM} USE_PCA=${USE_PCA}"
    echo "PCA_BASIS_DIR=${PCA_BASIS_DIR:-}"
    echo "SEED=${SEED}"
    echo "PCA_RANK=${PCA_RANK} PCA_SAMPLES=${PCA_SAMPLES} PCA_IMAGES_DIR=${PCA_IMAGES_DIR:-}"
    echo "============================================"
} > "${LOG_FILE}"

echo "============================================"
echo "当前实验唯一标识 (EXP_ID): $EXP_ID"
echo "MLP子空间: $USE_MLP (dim=${MLP_HIDDEN_DIM})"
echo "固定PCA: $USE_PCA (basis_dir=${PCA_BASIS_DIR:-N/A})"
echo "自动评测: $RUN_EVAL (gpus=${EVAL_GPUS}, num_gpus=${EVAL_NUM_GPUS}, tasks=${EVAL_TASKS:-all})"
echo "检查点路径: $CKPT_HOME"
echo "训练日志: $LOG_FILE"
echo "评测日志目录: $EVAL_LOG_DIR"
echo "合并日志目录: $MERGE_LOG_DIR"
echo "合并模型根目录: $MERGE_BASE_DIR"
echo "============================================"

# -------------------------------
# 4. 引擎与模型配置
# -------------------------------
TRAIN_FILES="${PROJECT_ROOT}/data/train.fall.parquet,${PROJECT_ROOT}/data/train.fight.fixed.parquet,${PROJECT_ROOT}/data/train.smoke.parquet"
FILES_JOINED="[${TRAIN_FILES}]"
BACKEND=${BACKEND:-fsdp}

if [[ "$BACKEND" == "fsdp" ]]; then
    ENGINE_CONFIG="engine=fsdp optim=fsdp optim.lr=2e-5 optim.lr_warmup_steps_ratio=0.01 optim.weight_decay=0.1 optim.betas=[0.9,0.95] optim.clip_grad=1.0 optim.min_lr_ratio=0.1 optim.warmup_style=cosine engine.ulysses_sequence_parallel_size=${SP_SIZE:-1} engine.strategy=${FSDP_STRATEGY:-fsdp2} engine.fsdp_size=${FSDP_SIZE:--1} engine.model_dtype=bfloat16 engine.dtype=bfloat16 engine.use_orig_params=True"
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
fi

if [[ "$USE_PCA" == true ]]; then
    safe_loc="${pos//\//__}"
    shopt -s nullglob
    single_basis=("${PCA_BASIS_DIR}/${safe_loc}.pt")
    multi_basis=("${PCA_BASIS_DIR}/${safe_loc}"__dim*.pt)
    shopt -u nullglob

    if [[ ! -f "${single_basis[0]}" && ${#multi_basis[@]} -eq 0 ]]; then
        {
            echo ""
            echo ">>> 未检测到 PCA basis .pt，自动执行 fit_pca_basis.py ..."
            echo ">>> PCA_BASIS_DIR=${PCA_BASIS_DIR}"
        } | tee -a "${LOG_FILE}"
        mkdir -p "${PCA_BASIS_DIR}"
        FIT_CMD=(
            "${UV_RUN[@]}" python "${PROJECT_ROOT}/scripts/fit_pca_basis.py"
            --model "$MODEL_ID"
            --locations "$pos"
            --rank "$PCA_RANK"
            --samples "$PCA_SAMPLES"
            --seed "$SEED"
            --train-files "$TRAIN_FILES"
            --out "$PCA_BASIS_DIR"
        )
        if [[ -n "$PCA_IMAGES_DIR" ]]; then
            FIT_CMD+=(--images-dir "$PCA_IMAGES_DIR")
        fi
        "${FIT_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
        {
            echo ">>> PCA basis 拟合完成: ${PCA_BASIS_DIR}"
        } | tee -a "${LOG_FILE}"
    else
        {
            echo ""
            echo ">>> 检测到已有 PCA basis，跳过拟合: ${PCA_BASIS_DIR}"
        } | tee -a "${LOG_FILE}"
    fi
fi

# -------------------------------
# 5. 执行训练
# -------------------------------
SAVE_FREQ="after_each_epoch"
echo "启动训练..."
{
    echo ""
    echo ">>> 启动 torchrun 训练 $(date -Is)"
} >> "${LOG_FILE}"

read -r -a ENGINE_ARGS <<< "${ENGINE_CONFIG}"
read -r -a MODEL_ARGS <<< "${MODEL_PARAMS}"
TORCHRUN_CMD=(
    "${UV_RUN[@]}" torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_TRAINERS:-8}"
    ${ENTRYPOINT}
    "data.train_files=${FILES_JOINED}"
    data.train_batch_size=128
    data.micro_batch_size_per_gpu=4
    data.max_length=2048
    data.use_dynamic_bsz=False
    "${ENGINE_ARGS[@]}"
    "${MODEL_ARGS[@]}"
    trainer.test_freq=-1
    "trainer.save_freq=${SAVE_FREQ}"
    trainer.logger=[console,wandb]
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXP_ID}"
    "trainer.seed=${SEED}"
    trainer.total_epochs=5
    "trainer.default_local_dir=${CKPT_HOME}"
    checkpoint.save_contents=[model,optimizer,extra]
    data.num_workers=0
    "$@"
)
"${TORCHRUN_CMD[@]}" >> "${LOG_FILE}" 2>&1 &

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
        TARGET_DIR="${MERGE_BASE_DIR}/${EXP_ID}_step${STEP}"

        if [ -f "${TARGET_DIR}/config.json" ]; then
            echo "-> Step ${STEP} 已经合并过了，跳过。"
            exit 0
        fi

        echo "-> 正在合并 Step: ${STEP} 到 ${TARGET_DIR}"
        mkdir -p "${TARGET_DIR}"

        "${UV_RUN[@]}" python "${PROJECT_ROOT}/scripts/training/legacy_model_merger.py" merge \
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

# -------------------------------
# 7. 评测（默认开启，仅评测最新 merged 结果）
# -------------------------------
if [[ "$RUN_EVAL" == true ]]; then
    shopt -s nullglob
    MERGED_DIRS=("${MERGE_BASE_DIR}/${EXP_ID}_step"*)
    shopt -u nullglob

    LAST_MERGED_STEP=-1
    LATEST_MERGED_DIR=""
    for d in "${MERGED_DIRS[@]}"; do
        [[ -d "$d" ]] || continue
        step="${d##*_step}"
        if [[ "$step" =~ ^[0-9]+$ ]] && (( step > LAST_MERGED_STEP )); then
            LAST_MERGED_STEP=$step
            LATEST_MERGED_DIR="$d"
        fi
    done

    if (( LAST_MERGED_STEP < 0 )) || [[ -z "$LATEST_MERGED_DIR" ]]; then
        echo "错误: 在 merge 输出目录未找到可评测模型: ${MERGE_BASE_DIR}/${EXP_ID}_step*" >&2
        exit 1
    fi

    EVAL_LOG_FILE="${EVAL_LOG_DIR}/eval_latest_step${LAST_MERGED_STEP}.log"
    echo ">>> 开始评测 latest merged step=${LAST_MERGED_STEP}"
    echo ">>> 模型路径: ${LATEST_MERGED_DIR}"
    echo ">>> 评测日志: ${EVAL_LOG_FILE}"

    EVAL_CMD=(
        "${UV_RUN[@]}" python "${PROJECT_ROOT}/scripts/evaluation/eval_qwen3_vl_models.py"
        --model_path_override "${LATEST_MERGED_DIR}"
        --test_fall_path "${PROJECT_ROOT}/data/test.fall.parquet"
        --test_smoke_path "${PROJECT_ROOT}/data/test.smoke.parquet"
        --test_fight_path "${PROJECT_ROOT}/data/test.fight.fixed.parquet"
        --eval_mode structured
        --prompt_mode dataset
        --batch_size_falldown "${EVAL_BATCH_FALLDOWN}"
        --batch_size_smoke "${EVAL_BATCH_SMOKE}"
        --batch_size_fight "${EVAL_BATCH_FIGHT}"
        --num_gpus "${EVAL_NUM_GPUS}"
        --seed "${SEED}"
        --dtype bfloat16
    )
    if [[ "$MODE" != "baseline" ]]; then
        EVAL_CMD+=(--external_lib vision_tower_dp_everywhere)
    fi
    if [[ -n "${EVAL_TASKS}" ]]; then
        EVAL_CMD+=(--tasks "${EVAL_TASKS}")
    fi

    CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" "${EVAL_CMD[@]}" > "${EVAL_LOG_FILE}" 2>&1
    echo ">>> 评测完成，日志: ${EVAL_LOG_FILE}"
else
    echo ">>> 跳过评测（--skip-eval）"
fi

echo "本次实验 (${EXP_ID}) 全流程结束！"

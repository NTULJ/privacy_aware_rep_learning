#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SUITE_TAG="${SUITE_TAG:-privacy_tradeoff_suite_$(date +%Y%m%d_%H%M%S)}"
SUITE_ROOT="${SUITE_ROOT:-$ROOT_DIR/runs/suites/$SUITE_TAG}"
LOG_DIR="$SUITE_ROOT/logs"
mkdir -p "$LOG_DIR"

MODEL_ID="${MODEL_ID:-$ROOT_DIR/models/Qwen3-VL-8B-Instruct}"
STANFORD_DATASET_ROOT="${STANFORD_DATASET_ROOT:-$ROOT_DIR/data/Stanford40/images}"
LFW_DATASET_ROOT="${LFW_DATASET_ROOT:-$ROOT_DIR/data/lfw}"
LFW_PAIR_FILE="${LFW_PAIR_FILE:-$ROOT_DIR/data/lfw_pairs.csv}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
DTYPE="${DTYPE:-bfloat16}"
EPSILON="${EPSILON:-64.0}"
DELTA_PRIV="${DELTA_PRIV:-1e-5}"
DELTA_MASK="${DELTA_MASK:-0.05}"
CLIP_NORM="${CLIP_NORM:-32.0}"
PATCH_ALPHA="${PATCH_ALPHA:-0.70}"
UPPER_BODY_WEIGHT="${UPPER_BODY_WEIGHT:-0.64}"
PERSON_MODEL="${PERSON_MODEL:-yolo11n.pt}"
FACE_MODEL="${FACE_MODEL:-}"
FACE_MODEL_KIND="${FACE_MODEL_KIND:-face}"
FACE_CONF="${FACE_CONF:-0.20}"
PERSON_CONF="${PERSON_CONF:-0.25}"
YUNET_MODEL="${YUNET_MODEL:-$ROOT_DIR/models/face_detection_yunet_2023mar.onnx}"
YUNET_SCORE_THRESHOLD="${YUNET_SCORE_THRESHOLD:-0.55}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-300}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
TRAIN_LR="${TRAIN_LR:-1e-2}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-1e-4}"
SEED="${SEED:-42}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
SAVE_TOKENS="${SAVE_TOKENS:-0}"
KEEP_SHARDS="${KEEP_SHARDS:-1}"
OURS_NOISE_SCALE_MULTIPLIER="${OURS_NOISE_SCALE_MULTIPLIER:-0.05}"
UNIFORM_NOISE_SCALE_MULTIPLIER="${UNIFORM_NOISE_SCALE_MULTIPLIER:-1.0}"
STAGE_NOISE_SCALE_MULTIPLIER="${STAGE_NOISE_SCALE_MULTIPLIER:-0.05}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
LOG_UPDATE_INTERVAL="${LOG_UPDATE_INTERVAL:-20}"
EXTRA_EXTRACT_ARGS="${EXTRA_EXTRACT_ARGS:-}"
EXTRA_PROBE_ARGS="${EXTRA_PROBE_ARGS:-}"
STAGE_RUN_ONLY="${STAGE_RUN_ONLY:-}"
TRADEOFF_DIR="${TRADEOFF_DIR:-$SUITE_ROOT/tradeoff}"
TRADEOFF_STANFORD_METHOD_METRIC="${TRADEOFF_STANFORD_METHOD_METRIC:-results.hpool_priv_global.test.accuracy}"
TRADEOFF_LFW_METHOD_METRIC="${TRADEOFF_LFW_METHOD_METRIC:-results.priv_verification.test.auc}"
TRADEOFF_STANFORD_STAGE_METRIC="${TRADEOFF_STANFORD_STAGE_METRIC:-hpool_priv_global_test_acc}"
TRADEOFF_LFW_STAGE_METRIC="${TRADEOFF_LFW_STAGE_METRIC:-priv_verification_test_auc}"
TRADEOFF_TITLE_PREFIX="${TRADEOFF_TITLE_PREFIX:-Stanford40 vs LFW}"

RUN_STANFORD_OURS="${RUN_STANFORD_OURS:-1}"
RUN_STANFORD_UNIFORM="${RUN_STANFORD_UNIFORM:-1}"
RUN_STANFORD_STAGE="${RUN_STANFORD_STAGE:-1}"
RUN_LFW_OURS="${RUN_LFW_OURS:-1}"
RUN_LFW_UNIFORM="${RUN_LFW_UNIFORM:-1}"
RUN_LFW_STAGE="${RUN_LFW_STAGE:-1}"
PLOT_TRADEOFF="${PLOT_TRADEOFF:-1}"

STANFORD_OURS_OUTPUT_TAG="${STANFORD_OURS_OUTPUT_TAG:-${SUITE_TAG}_stanford_ours}"
STANFORD_UNIFORM_OUTPUT_TAG="${STANFORD_UNIFORM_OUTPUT_TAG:-${SUITE_TAG}_stanford_uniform}"
STANFORD_STAGE_OUTPUT_TAG="${STANFORD_STAGE_OUTPUT_TAG:-${SUITE_TAG}_stanford_stage}"
LFW_OURS_OUTPUT_TAG="${LFW_OURS_OUTPUT_TAG:-${SUITE_TAG}_lfw_ours}"
LFW_UNIFORM_OUTPUT_TAG="${LFW_UNIFORM_OUTPUT_TAG:-${SUITE_TAG}_lfw_uniform}"
LFW_STAGE_OUTPUT_TAG="${LFW_STAGE_OUTPUT_TAG:-${SUITE_TAG}_lfw_stage}"

run_with_log() {
  local name="$1"
  shift
  local log_path="$LOG_DIR/${name}.log"
  echo "============================================================" | tee "$log_path"
  echo "[suite] $name" | tee -a "$log_path"
  echo "  log: $log_path" | tee -a "$log_path"
  echo "============================================================" | tee -a "$log_path"
  "$@" 2>&1 | tee -a "$log_path"
}

common_env=(
  PYTHON_BIN="$PYTHON_BIN"
  MODEL_ID="$MODEL_ID"
  GPU_IDS="$GPU_IDS"
  PROCS_PER_GPU="$PROCS_PER_GPU"
  DTYPE="$DTYPE"
  EPSILON="$EPSILON"
  DELTA_PRIV="$DELTA_PRIV"
  DELTA_MASK="$DELTA_MASK"
  CLIP_NORM="$CLIP_NORM"
  PATCH_ALPHA="$PATCH_ALPHA"
  UPPER_BODY_WEIGHT="$UPPER_BODY_WEIGHT"
  PERSON_MODEL="$PERSON_MODEL"
  FACE_MODEL="$FACE_MODEL"
  FACE_MODEL_KIND="$FACE_MODEL_KIND"
  FACE_CONF="$FACE_CONF"
  PERSON_CONF="$PERSON_CONF"
  YUNET_MODEL="$YUNET_MODEL"
  YUNET_SCORE_THRESHOLD="$YUNET_SCORE_THRESHOLD"
  TRAIN_EPOCHS="$TRAIN_EPOCHS"
  TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE"
  TRAIN_LR="$TRAIN_LR"
  TRAIN_WEIGHT_DECAY="$TRAIN_WEIGHT_DECAY"
  SEED="$SEED"
  LOCAL_FILES_ONLY="$LOCAL_FILES_ONLY"
  SAVE_TOKENS="$SAVE_TOKENS"
  KEEP_SHARDS="$KEEP_SHARDS"
  PYTHONUNBUFFERED="$PYTHONUNBUFFERED"
  LOG_UPDATE_INTERVAL="$LOG_UPDATE_INTERVAL"
  EXTRA_EXTRACT_ARGS="$EXTRA_EXTRACT_ARGS"
  EXTRA_PROBE_ARGS="$EXTRA_PROBE_ARGS"
)

if [[ "$RUN_STANFORD_OURS" == "1" ]]; then
  run_with_log "stanford_ours" env \
    "${common_env[@]}" \
    OUTPUT_TAG="$STANFORD_OURS_OUTPUT_TAG" \
    DATASET_ROOT="$STANFORD_DATASET_ROOT" \
    NOISE_SCALE_MULTIPLIER="$OURS_NOISE_SCALE_MULTIPLIER" \
    bash "$ROOT_DIR/run_stanford40_full_eval_multigpu.sh"
fi

if [[ "$RUN_STANFORD_UNIFORM" == "1" ]]; then
  run_with_log "stanford_uniform" env \
    "${common_env[@]}" \
    OUTPUT_TAG="$STANFORD_UNIFORM_OUTPUT_TAG" \
    DATASET_ROOT="$STANFORD_DATASET_ROOT" \
    NOISE_SCALE_MULTIPLIER="$UNIFORM_NOISE_SCALE_MULTIPLIER" \
    bash "$ROOT_DIR/run_stanford40_full_eval_multigpu_uniform_noise.sh"
fi

if [[ "$RUN_STANFORD_STAGE" == "1" ]]; then
  run_with_log "stanford_stage" env \
    "${common_env[@]}" \
    BASE_OUTPUT_TAG="$STANFORD_STAGE_OUTPUT_TAG" \
    DATASET_ROOT="$STANFORD_DATASET_ROOT" \
    RUN_ONLY="$STAGE_RUN_ONLY" \
    NOISE_SCALE_MULTIPLIER="$STAGE_NOISE_SCALE_MULTIPLIER" \
    bash "$ROOT_DIR/run_stanford40_full_eval_multigpu_stage_injection.sh"
fi

if [[ "$RUN_LFW_OURS" == "1" ]]; then
  run_with_log "lfw_ours" env \
    "${common_env[@]}" \
    OUTPUT_TAG="$LFW_OURS_OUTPUT_TAG" \
    DATASET_ROOT="$LFW_DATASET_ROOT" \
    PAIR_FILE="$LFW_PAIR_FILE" \
    NOISE_SCALE_MULTIPLIER="$OURS_NOISE_SCALE_MULTIPLIER" \
    bash "$ROOT_DIR/run_lfw_full_eval_multigpu.sh"
fi

if [[ "$RUN_LFW_UNIFORM" == "1" ]]; then
  run_with_log "lfw_uniform" env \
    "${common_env[@]}" \
    OUTPUT_TAG="$LFW_UNIFORM_OUTPUT_TAG" \
    DATASET_ROOT="$LFW_DATASET_ROOT" \
    PAIR_FILE="$LFW_PAIR_FILE" \
    NOISE_SCALE_MULTIPLIER="$UNIFORM_NOISE_SCALE_MULTIPLIER" \
    bash "$ROOT_DIR/run_lfw_full_eval_multigpu_uniform_noise.sh"
fi

if [[ "$RUN_LFW_STAGE" == "1" ]]; then
  run_with_log "lfw_stage" env \
    "${common_env[@]}" \
    BASE_OUTPUT_TAG="$LFW_STAGE_OUTPUT_TAG" \
    DATASET_ROOT="$LFW_DATASET_ROOT" \
    PAIR_FILE="$LFW_PAIR_FILE" \
    RUN_ONLY="$STAGE_RUN_ONLY" \
    NOISE_SCALE_MULTIPLIER="$STAGE_NOISE_SCALE_MULTIPLIER" \
    bash "$ROOT_DIR/run_lfw_full_eval_multigpu_stage_injection.sh"
fi

STANFORD_OURS_SUMMARY="$ROOT_DIR/runs/probes/$STANFORD_OURS_OUTPUT_TAG/summary.json"
STANFORD_UNIFORM_SUMMARY="$ROOT_DIR/runs/probes/$STANFORD_UNIFORM_OUTPUT_TAG/summary.json"
STANFORD_STAGE_SUMMARY="$ROOT_DIR/runs/sweeps/$STANFORD_STAGE_OUTPUT_TAG/stage_sweep_summary.json"
LFW_OURS_SUMMARY="$ROOT_DIR/runs/probes/$LFW_OURS_OUTPUT_TAG/summary.json"
LFW_UNIFORM_SUMMARY="$ROOT_DIR/runs/probes/$LFW_UNIFORM_OUTPUT_TAG/summary.json"
LFW_STAGE_SUMMARY="$ROOT_DIR/runs/sweeps/$LFW_STAGE_OUTPUT_TAG/stage_sweep_summary.json"

if [[ "$PLOT_TRADEOFF" == "1" ]]; then
  run_with_log "plot_tradeoff" "$PYTHON_BIN" "$ROOT_DIR/scripts/plot_privacy_utility_tradeoff.py" \
    --stanford-ours-summary "$STANFORD_OURS_SUMMARY" \
    --lfw-ours-summary "$LFW_OURS_SUMMARY" \
    --stanford-uniform-summary "$STANFORD_UNIFORM_SUMMARY" \
    --lfw-uniform-summary "$LFW_UNIFORM_SUMMARY" \
    --stanford-stage-summary "$STANFORD_STAGE_SUMMARY" \
    --lfw-stage-summary "$LFW_STAGE_SUMMARY" \
    --output-dir "$TRADEOFF_DIR" \
    --stanford-method-metric "$TRADEOFF_STANFORD_METHOD_METRIC" \
    --lfw-method-metric "$TRADEOFF_LFW_METHOD_METRIC" \
    --stanford-stage-metric "$TRADEOFF_STANFORD_STAGE_METRIC" \
    --lfw-stage-metric "$TRADEOFF_LFW_STAGE_METRIC" \
    --title-prefix "$TRADEOFF_TITLE_PREFIX"
fi

echo
echo "suite done"
echo "suite_root:               $SUITE_ROOT"
echo "stanford_ours_summary:    $STANFORD_OURS_SUMMARY"
echo "stanford_uniform_summary: $STANFORD_UNIFORM_SUMMARY"
echo "stanford_stage_summary:   $STANFORD_STAGE_SUMMARY"
echo "lfw_ours_summary:         $LFW_OURS_SUMMARY"
echo "lfw_uniform_summary:      $LFW_UNIFORM_SUMMARY"
echo "lfw_stage_summary:        $LFW_STAGE_SUMMARY"
echo "tradeoff_dir:             $TRADEOFF_DIR"

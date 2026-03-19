#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_OUTPUT_TAG="${BASE_OUTPUT_TAG:-lfw_full_mgpu_stage_injection_$(date +%Y%m%d_%H%M%S)}"
RUN_ONLY="${RUN_ONLY:-}"

DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/data/lfw}"
PAIR_FILE="${PAIR_FILE:-$ROOT_DIR/data/lfw_pairs.csv}"
MODEL_ID="${MODEL_ID:-$ROOT_DIR/models/Qwen3-VL-8B-Instruct}"
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
TRAIN_EPOCHS="${TRAIN_EPOCHS:-120}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
TRAIN_LR="${TRAIN_LR:-1e-2}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-1e-4}"
SEED="${SEED:-42}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
SAVE_TOKENS="${SAVE_TOKENS:-0}"
EXTRA_EXTRACT_ARGS="${EXTRA_EXTRACT_ARGS:-}"
EXTRA_PROBE_ARGS="${EXTRA_PROBE_ARGS:-}"
KEEP_SHARDS="${KEEP_SHARDS:-1}"
NOISE_SCALE_MULTIPLIER="${NOISE_SCALE_MULTIPLIER:-0.05}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
LOG_UPDATE_INTERVAL="${LOG_UPDATE_INTERVAL:-20}"
VERIFICATION_FEATURE_CLEAN="${VERIFICATION_FEATURE_CLEAN:-hpool_clean__head_mean}"
VERIFICATION_FEATURE_PRIV="${VERIFICATION_FEATURE_PRIV:-hpool_priv__head_mean}"
VERIFICATION_METRIC="${VERIFICATION_METRIC:-cosine}"
FAR_TARGET="${FAR_TARGET:-0.01}"

SWEEP_ROOT="${SWEEP_ROOT:-$ROOT_DIR/runs/sweeps/$BASE_OUTPUT_TAG}"
SUMMARY_DIR="$SWEEP_ROOT/summaries"
LOG_DIR="$SWEEP_ROOT/logs"
mkdir -p "$SUMMARY_DIR" "$LOG_DIR"

INJECTION_STAGES=(x_pre block1 block8 block16 block24 hpool)

should_run() {
  local stage="$1"
  if [[ -z "$RUN_ONLY" ]]; then
    return 0
  fi
  IFS=',' read -r -a requested <<< "$RUN_ONLY"
  for item in "${requested[@]}"; do
    if [[ "$item" == "$stage" ]]; then
      return 0
    fi
  done
  return 1
}

for injection_stage in "${INJECTION_STAGES[@]}"; do
  if ! should_run "$injection_stage"; then
    continue
  fi

  output_tag="${BASE_OUTPUT_TAG}_${injection_stage}"
  log_path="$LOG_DIR/${injection_stage}.log"

  echo "============================================================" | tee "$log_path"
  echo "[run] injection_stage=$injection_stage" | tee -a "$log_path"
  echo "  output_tag=$output_tag" | tee -a "$log_path"
  echo "============================================================" | tee -a "$log_path"

  env \
    PYTHON_BIN="$PYTHON_BIN" \
    EXTRACTOR_SCRIPT="$ROOT_DIR/scripts/extract_qwen_vl_features_stage_injection.py" \
    OUTPUT_TAG="$output_tag" \
    MODEL_ID="$MODEL_ID" \
    DATASET_ROOT="$DATASET_ROOT" \
    PAIR_FILE="$PAIR_FILE" \
    GPU_IDS="$GPU_IDS" \
    PROCS_PER_GPU="$PROCS_PER_GPU" \
    DTYPE="$DTYPE" \
    EPSILON="$EPSILON" \
    DELTA_PRIV="$DELTA_PRIV" \
    DELTA_MASK="$DELTA_MASK" \
    CLIP_NORM="$CLIP_NORM" \
    PATCH_ALPHA="$PATCH_ALPHA" \
    UPPER_BODY_WEIGHT="$UPPER_BODY_WEIGHT" \
    PERSON_MODEL="$PERSON_MODEL" \
    FACE_MODEL="$FACE_MODEL" \
    FACE_MODEL_KIND="$FACE_MODEL_KIND" \
    FACE_CONF="$FACE_CONF" \
    PERSON_CONF="$PERSON_CONF" \
    YUNET_MODEL="$YUNET_MODEL" \
    YUNET_SCORE_THRESHOLD="$YUNET_SCORE_THRESHOLD" \
    TRAIN_EPOCHS="$TRAIN_EPOCHS" \
    TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    TRAIN_LR="$TRAIN_LR" \
    TRAIN_WEIGHT_DECAY="$TRAIN_WEIGHT_DECAY" \
    SEED="$SEED" \
    LOCAL_FILES_ONLY="$LOCAL_FILES_ONLY" \
    SAVE_TOKENS="$SAVE_TOKENS" \
    KEEP_SHARDS="$KEEP_SHARDS" \
    NOISE_SCALE_MULTIPLIER="$NOISE_SCALE_MULTIPLIER" \
    PYTHONUNBUFFERED="$PYTHONUNBUFFERED" \
    LOG_UPDATE_INTERVAL="$LOG_UPDATE_INTERVAL" \
    EXTRACT_STAGES="hpool_clean hpool_priv" \
    VERIFICATION_FEATURE_CLEAN="$VERIFICATION_FEATURE_CLEAN" \
    VERIFICATION_FEATURE_PRIV="$VERIFICATION_FEATURE_PRIV" \
    VERIFICATION_METRIC="$VERIFICATION_METRIC" \
    FAR_TARGET="$FAR_TARGET" \
    EXTRA_PROBE_ARGS="$EXTRA_PROBE_ARGS" \
    EXTRA_EXTRACT_ARGS="--injection-stage $injection_stage $EXTRA_EXTRACT_ARGS" \
    bash "$ROOT_DIR/run_scripts/run_lfw_full_eval_multigpu.sh" 2>&1 | tee -a "$log_path"

  summary_src="$ROOT_DIR/runs/probes/$output_tag/summary.json"
  summary_dst="$SUMMARY_DIR/${injection_stage}.json"
  if [[ ! -f "$summary_src" ]]; then
    echo "missing summary for $injection_stage: $summary_src" >&2
    exit 1
  fi

  export SUMMARY_SRC="$summary_src" SUMMARY_DST="$summary_dst" INJECTION_STAGE="$injection_stage"
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

src = Path(os.environ['SUMMARY_SRC'])
dst = Path(os.environ['SUMMARY_DST'])
payload = json.loads(src.read_text(encoding='utf-8'))
payload['injection_stage'] = os.environ['INJECTION_STAGE']
dst.write_text(json.dumps(payload, indent=2), encoding='utf-8')
print(f'wrote {dst}')
PY
done

export SWEEP_ROOT
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

sweep_root = Path(os.environ['SWEEP_ROOT'])
summary_dir = sweep_root / 'summaries'
out_json = sweep_root / 'stage_sweep_summary.json'
out_csv = sweep_root / 'stage_sweep_summary.csv'

records = []
for path in sorted(summary_dir.glob('*.json')):
    payload = json.loads(path.read_text(encoding='utf-8'))
    record = {
        'injection_stage': payload.get('injection_stage'),
        'summary_path': str(path),
        'noise_scale_multiplier': payload.get('noise_scale_multiplier'),
        'clip_norm': payload.get('clip_norm'),
    }
    clean_metrics = payload.get('results', {}).get('clean_verification', {})
    priv_metrics = payload.get('results', {}).get('priv_verification', {})
    for prefix, metrics in [('clean_verification', clean_metrics), ('priv_verification', priv_metrics)]:
        for split in ['train', 'val', 'test']:
            split_metrics = metrics.get(split, {})
            record[f'{prefix}_{split}_accuracy'] = split_metrics.get('accuracy')
            record[f'{prefix}_{split}_auc'] = split_metrics.get('auc')
            record[f'{prefix}_{split}_tar_at_far'] = split_metrics.get('tar_at_far')
    records.append(record)

out_json.write_text(json.dumps(records, indent=2), encoding='utf-8')
headers = [
    'injection_stage',
    'summary_path',
    'noise_scale_multiplier',
    'clip_norm',
    'clean_verification_test_accuracy',
    'clean_verification_test_auc',
    'clean_verification_test_tar_at_far',
    'priv_verification_test_accuracy',
    'priv_verification_test_auc',
    'priv_verification_test_tar_at_far',
]
with out_csv.open('w', encoding='utf-8', newline='') as handle:
    handle.write(','.join(headers) + '\n')
    for record in records:
        handle.write(','.join('' if record.get(h) is None else str(record.get(h)) for h in headers) + '\n')

print(json.dumps(records, indent=2))
print(f'stage sweep summary json: {out_json}')
print(f'stage sweep summary csv:  {out_csv}')
PY

echo
echo "done"
echo "sweep_root: $SWEEP_ROOT"
echo "json:       $SWEEP_ROOT/stage_sweep_summary.json"
echo "csv:        $SWEEP_ROOT/stage_sweep_summary.csv"

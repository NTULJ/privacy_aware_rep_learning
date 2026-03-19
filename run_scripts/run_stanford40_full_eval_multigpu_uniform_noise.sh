#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
EXTRACTOR_SCRIPT="${EXTRACTOR_SCRIPT:-$ROOT_DIR/scripts/extract_qwen_vl_features_uniform_noise.py}"
MODEL_ID="${MODEL_ID:-$ROOT_DIR/models/Qwen3-VL-8B-Instruct}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/data/Stanford40/images}"
OUTPUT_TAG="${OUTPUT_TAG:-stanford40_full_mgpu_uniform_$(date +%Y%m%d_%H%M%S)}"
WORK_ROOT="${WORK_ROOT:-$ROOT_DIR/runs/mgpu/$OUTPUT_TAG}"
OUTPUT_FEATURE_ROOT="${OUTPUT_FEATURE_ROOT:-$ROOT_DIR/runs/features/$OUTPUT_TAG}"
OUTPUT_PROBE_ROOT="${OUTPUT_PROBE_ROOT:-$ROOT_DIR/runs/probes/$OUTPUT_TAG}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
DEVICE_DTYPE="${DTYPE:-bfloat16}"
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
EXTRA_EXTRACT_ARGS="${EXTRA_EXTRACT_ARGS:-}"
EXTRA_PROBE_ARGS="${EXTRA_PROBE_ARGS:-}"
KEEP_SHARDS="${KEEP_SHARDS:-1}"
NOISE_SCALE_MULTIPLIER="${NOISE_SCALE_MULTIPLIER:-0.05}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
LOG_UPDATE_INTERVAL="${LOG_UPDATE_INTERVAL:-20}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
SHARD_COUNT=$(( ${#GPU_ARRAY[@]} * PROCS_PER_GPU ))
if [[ "$SHARD_COUNT" -le 0 ]]; then
  echo "invalid SHARD_COUNT=$SHARD_COUNT" >&2
  exit 1
fi

MASTER_MANIFEST="$WORK_ROOT/master_manifest.jsonl"
SHARD_MANIFEST_DIR="$WORK_ROOT/manifests"
SHARD_OUTPUT_DIR="$WORK_ROOT/shards"
LOG_DIR="$WORK_ROOT/logs"
mkdir -p "$WORK_ROOT" "$SHARD_MANIFEST_DIR" "$SHARD_OUTPUT_DIR" "$LOG_DIR" "$OUTPUT_FEATURE_ROOT" "$OUTPUT_PROBE_ROOT"

export ROOT_DIR DATASET_ROOT MASTER_MANIFEST SHARD_MANIFEST_DIR SHARD_COUNT
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ['DATASET_ROOT'])
master_manifest = Path(os.environ['MASTER_MANIFEST'])
manifest_dir = Path(os.environ['SHARD_MANIFEST_DIR'])
shard_count = int(os.environ['SHARD_COUNT'])

images = sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}])
records = []
for path in images:
    stem = path.stem
    label = stem.rsplit('_', 1)[0] if '_' in stem else stem
    records.append({
        'sample_id': stem,
        'image_path': str(path.resolve()),
        'split': 'all',
        'label': label,
        'dataset': 'stanford40',
        'person_id': '',
    })
master_manifest.parent.mkdir(parents=True, exist_ok=True)
with master_manifest.open('w', encoding='utf-8') as handle:
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=True) + '\n')

shards = [[] for _ in range(shard_count)]
for idx, record in enumerate(records):
    shards[idx % shard_count].append(record)

manifest_dir.mkdir(parents=True, exist_ok=True)
for shard_idx, shard_records in enumerate(shards):
    shard_path = manifest_dir / f'shard_{shard_idx:02d}.jsonl'
    with shard_path.open('w', encoding='utf-8') as handle:
        for record in shard_records:
            handle.write(json.dumps(record, ensure_ascii=True) + '\n')
print(f'prepared {len(records)} samples into {shard_count} shards at {manifest_dir}')
PY

run_shard() {
  local shard_idx="$1"
  local gpu_slot="$2"
  local gpu_id="$3"
  local shard_manifest="$SHARD_MANIFEST_DIR/shard_$(printf '%02d' "$shard_idx").jsonl"
  local shard_output="$SHARD_OUTPUT_DIR/shard_$(printf '%02d' "$shard_idx")"
  local shard_log="$LOG_DIR/shard_$(printf '%02d' "$shard_idx").log"

  mkdir -p "$shard_output"
  echo "[extract shard=$shard_idx gpu=$gpu_id] -> $shard_output"

  extract_cmd=(
    "$PYTHON_BIN" "$EXTRACTOR_SCRIPT"
    --manifest "$shard_manifest"
    --model-id "$MODEL_ID"
    --mode vision-only
    --output "$shard_output"
    --device cuda:0
    --dtype "$DEVICE_DTYPE"
    --epsilon "$EPSILON"
    --delta-priv "$DELTA_PRIV"
    --delta-mask "$DELTA_MASK"
    --noise-scale-multiplier "$NOISE_SCALE_MULTIPLIER"
    --clip-norm "$CLIP_NORM"
    --patch-alpha "$PATCH_ALPHA"
    --upper-body-weight "$UPPER_BODY_WEIGHT"
    --person-model "$PERSON_MODEL"
    --person-conf "$PERSON_CONF"
    --yunet-model "$YUNET_MODEL"
    --yunet-score-threshold "$YUNET_SCORE_THRESHOLD"
    --stages hpool_clean hpool_priv
  )

  if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
    extract_cmd+=(--local-files-only)
  fi
  if [[ "$SAVE_TOKENS" == "1" ]]; then
    extract_cmd+=(--save-token-features)
  fi
  if [[ -n "$FACE_MODEL" ]]; then
    extract_cmd+=(--face-model "$FACE_MODEL" --face-model-kind "$FACE_MODEL_KIND" --face-conf "$FACE_CONF")
  fi
  if [[ -n "$EXTRA_EXTRACT_ARGS" ]]; then
    # shellcheck disable=SC2206
    extra_extract=( $EXTRA_EXTRACT_ARGS )
    extract_cmd+=("${extra_extract[@]}")
  fi

  PYTHONUNBUFFERED="$PYTHONUNBUFFERED" CUDA_VISIBLE_DEVICES="$gpu_id" "${extract_cmd[@]}" > "$shard_log" 2>&1
}

monitor_shards() {
  while true; do
    local alive=0
    echo "[progress $(date +%H:%M:%S)]"
    for ((idx=0; idx<SHARD_COUNT; idx++)); do
      local log_path="$LOG_DIR/shard_$(printf '%02d' "$idx").log"
      if [[ -f "$log_path" ]]; then
        local last_line
        last_line=$(tail -n 1 "$log_path" 2>/dev/null || true)
        if [[ -z "$last_line" ]]; then
          echo "  shard $(printf '%02d' "$idx"): started"
        else
          echo "  shard $(printf '%02d' "$idx"): $last_line"
        fi
      else
        echo "  shard $(printf '%02d' "$idx"): pending"
      fi
    done

    for pid in "${pids[@]:-}"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive=1
        break
      fi
    done

    if [[ "$alive" -eq 0 ]]; then
      break
    fi
    sleep "$LOG_UPDATE_INTERVAL"
  done
}

pids=()
shard_idx=0
for gpu_id in "${GPU_ARRAY[@]}"; do
  for ((slot=0; slot<PROCS_PER_GPU; slot++)); do
    if [[ "$shard_idx" -ge "$SHARD_COUNT" ]]; then
      break
    fi
    run_shard "$shard_idx" "$slot" "$gpu_id" &
    pids+=("$!")
    shard_idx=$((shard_idx + 1))
  done
done

monitor_shards &
monitor_pid=$!

extract_failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    extract_failed=1
  fi
done

if kill -0 "$monitor_pid" 2>/dev/null; then
  kill "$monitor_pid" 2>/dev/null || true
fi
wait "$monitor_pid" 2>/dev/null || true

if [[ "$extract_failed" -ne 0 ]]; then
  echo "one or more shard extractions failed; inspect $LOG_DIR" >&2
  exit 1
fi

echo "all shard extractions finished"

export SHARD_OUTPUT_DIR SHARD_COUNT OUTPUT_FEATURE_ROOT SAVE_TOKENS
"$PYTHON_BIN" - <<'PY'
import csv
import json
import os
from pathlib import Path
import numpy as np

shard_root = Path(os.environ['SHARD_OUTPUT_DIR'])
out_root = Path(os.environ['OUTPUT_FEATURE_ROOT'])
shard_count = int(os.environ['SHARD_COUNT'])
save_tokens = os.environ['SAVE_TOKENS'] == '1'

out_root.mkdir(parents=True, exist_ok=True)
(out_root / 'pooled').mkdir(parents=True, exist_ok=True)
(out_root / 'patch').mkdir(parents=True, exist_ok=True)
(out_root / 'stats').mkdir(parents=True, exist_ok=True)
if save_tokens:
    (out_root / 'tokens').mkdir(parents=True, exist_ok=True)

shard_dirs = [shard_root / f'shard_{idx:02d}' for idx in range(shard_count)]
for shard_dir in shard_dirs:
    if not shard_dir.exists():
        raise FileNotFoundError(f'missing shard output: {shard_dir}')

manifest_lines = []
label_rows = []
index_rows = []
summary = {
    'requested_samples': 0,
    'successful_samples': 0,
    'failed_samples': 0,
    'token_features_saved': save_tokens,
    'shards': [],
}
pooled_buffers = {}
config_payload = None

for shard_dir in shard_dirs:
    config_path = shard_dir / 'config.json'
    if config_path.exists() and config_payload is None:
        config_payload = json.loads(config_path.read_text(encoding='utf-8'))

    manifest_path = shard_dir / 'manifest.jsonl'
    if manifest_path.exists():
        manifest_lines.extend([line for line in manifest_path.read_text(encoding='utf-8').splitlines() if line.strip()])

    labels_path = shard_dir / 'labels.csv'
    if labels_path.exists():
        with labels_path.open('r', encoding='utf-8', newline='') as handle:
            label_rows.extend(list(csv.DictReader(handle)))

    index_path = shard_dir / 'index.csv'
    if index_path.exists():
        with index_path.open('r', encoding='utf-8', newline='') as handle:
            index_rows.extend(list(csv.DictReader(handle)))

    stats_path = shard_dir / 'stats' / 'extraction_summary.json'
    if stats_path.exists():
        shard_stats = json.loads(stats_path.read_text(encoding='utf-8'))
        summary['requested_samples'] += int(shard_stats.get('requested_samples', 0))
        summary['successful_samples'] += int(shard_stats.get('successful_samples', 0))
        summary['failed_samples'] += int(shard_stats.get('failed_samples', 0))
        summary['shards'].append({'name': shard_dir.name, **shard_stats})

    pooled_dir = shard_dir / 'pooled'
    for npy_path in sorted(pooled_dir.glob('*.npy')):
        pooled_buffers.setdefault(npy_path.name, []).append(np.load(npy_path))

    for patch_path in sorted((shard_dir / 'patch').glob('*.npz')):
        target = out_root / 'patch' / patch_path.name
        if not target.exists():
            os.link(patch_path, target)

    if save_tokens:
        for token_path in sorted((shard_dir / 'tokens').glob('*.npz')):
            target = out_root / 'tokens' / token_path.name
            if not target.exists():
                os.link(token_path, target)

if config_payload is not None:
    config_payload['merged_from_shards'] = shard_count
    config_payload['noise_distribution'] = 'uniform_global'
    config_payload['spatial_reweighting'] = False
    (out_root / 'config.json').write_text(json.dumps(config_payload, indent=2), encoding='utf-8')

(out_root / 'manifest.jsonl').write_text('\n'.join(manifest_lines) + ('\n' if manifest_lines else ''), encoding='utf-8')

with (out_root / 'labels.csv').open('w', encoding='utf-8', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=['sample_id', 'image_path', 'split', 'label', 'dataset', 'person_id'])
    writer.writeheader()
    for row in label_rows:
        writer.writerow(row)

with (out_root / 'index.csv').open('w', encoding='utf-8', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=['sample_id', 'image_path', 'split', 'label', 'dataset', 'person_id', 'token_path', 'patch_path'])
    writer.writeheader()
    for row in index_rows:
        writer.writerow(row)

for name, arrays in pooled_buffers.items():
    merged = np.concatenate(arrays, axis=0)
    np.save(out_root / 'pooled' / name, merged.astype(np.float32))

(out_root / 'stats' / 'extraction_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(f'merged shard outputs into {out_root}')
PY

run_probe() {
  local feature_key="$1"
  local run_name="$2"
  echo "[probe] $feature_key -> $OUTPUT_PROBE_ROOT/$run_name"
  probe_cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/scripts/train_linear_probe.py"
    --feature-root "$OUTPUT_FEATURE_ROOT"
    --feature-key "$feature_key"
    --task classification
    --output "$OUTPUT_PROBE_ROOT/$run_name"
    --standardize
    --class-weight balanced
    --epochs "$TRAIN_EPOCHS"
    --batch-size "$TRAIN_BATCH_SIZE"
    --lr "$TRAIN_LR"
    --weight-decay "$TRAIN_WEIGHT_DECAY"
    --seed "$SEED"
    --device cuda:0
  )
  if [[ -n "$EXTRA_PROBE_ARGS" ]]; then
    # shellcheck disable=SC2206
    extra_probe=( $EXTRA_PROBE_ARGS )
    probe_cmd+=("${extra_probe[@]}")
  fi
  "${probe_cmd[@]}"
}

echo "[probe 1/4] clean global"
run_probe "hpool_clean__global_mean" "hpool_clean_global"

echo "[probe 2/4] priv global"
run_probe "hpool_priv__global_mean" "hpool_priv_global"

echo "[probe 3/4] clean nonhead-human"
run_probe "hpool_clean__nonhead_human_mean" "hpool_clean_nonhead"

echo "[probe 4/4] priv nonhead-human"
run_probe "hpool_priv__nonhead_human_mean" "hpool_priv_nonhead"

SUMMARY_PATH="$OUTPUT_PROBE_ROOT/summary.json"
export SUMMARY_PATH OUTPUT_FEATURE_ROOT OUTPUT_PROBE_ROOT EPSILON DELTA_PRIV DELTA_MASK CLIP_NORM PATCH_ALPHA UPPER_BODY_WEIGHT GPU_IDS PROCS_PER_GPU DEVICE_DTYPE DATASET_ROOT MODEL_ID NOISE_SCALE_MULTIPLIER EXTRACTOR_SCRIPT
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

probe_root = Path(os.environ['OUTPUT_PROBE_ROOT'])
feature_root = Path(os.environ['OUTPUT_FEATURE_ROOT'])
summary = {
    'feature_root': str(feature_root),
    'probe_root': str(probe_root),
    'dataset_root': os.environ['DATASET_ROOT'],
    'model_id': os.environ['MODEL_ID'],
    'extractor_script': os.environ['EXTRACTOR_SCRIPT'],
    'epsilon': float(os.environ['EPSILON']),
    'delta_priv': float(os.environ['DELTA_PRIV']),
    'delta_mask': float(os.environ['DELTA_MASK']),
    'clip_norm': float(os.environ['CLIP_NORM']),
    'noise_scale_multiplier': float(os.environ['NOISE_SCALE_MULTIPLIER']),
    'noise_distribution': 'uniform_global',
    'spatial_reweighting': False,
    'patch_alpha': float(os.environ['PATCH_ALPHA']),
    'upper_body_weight': float(os.environ['UPPER_BODY_WEIGHT']),
    'gpu_ids': os.environ['GPU_IDS'],
    'procs_per_gpu': int(os.environ['PROCS_PER_GPU']),
    'dtype': os.environ['DEVICE_DTYPE'],
    'results': {},
}
for name in ['hpool_clean_global', 'hpool_priv_global', 'hpool_clean_nonhead', 'hpool_priv_nonhead']:
    metrics_path = probe_root / name / 'metrics.json'
    if metrics_path.exists():
        summary['results'][name] = json.loads(metrics_path.read_text(encoding='utf-8'))
summary_path = Path(os.environ['SUMMARY_PATH'])
summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
PY

if [[ "$KEEP_SHARDS" != "1" ]]; then
  rm -rf "$SHARD_OUTPUT_DIR" "$SHARD_MANIFEST_DIR"
fi

echo
echo "done"
echo "work_root:      $WORK_ROOT"
echo "feature_root:   $OUTPUT_FEATURE_ROOT"
echo "probe_root:     $OUTPUT_PROBE_ROOT"
echo "summary:        $SUMMARY_PATH"

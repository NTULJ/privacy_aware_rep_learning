#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
FEATURE_ROOT="${FEATURE_ROOT:-}"
if [[ -z "$FEATURE_ROOT" ]]; then
  echo "FEATURE_ROOT is required" >&2
  exit 1
fi

DATASET_NAME="${DATASET_NAME:-$(basename "$(dirname "$FEATURE_ROOT")")}"
FEATURE_VERSION="${FEATURE_VERSION:-$(basename "$FEATURE_ROOT")}" 
METHOD="${METHOD:-ours}"
STAGE="${STAGE:-hpool}"
FEATURE_KEY="${FEATURE_KEY:-hpool_priv__global_mean}"

OUTPUT_PROBE_ROOT="${OUTPUT_PROBE_ROOT:-$ROOT_DIR/runs/probes/$DATASET_NAME/$FEATURE_VERSION/utility_${METHOD}_${STAGE}}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/probes}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"

LABEL_KEY="${LABEL_KEY:-label}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-300}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
TRAIN_LR="${TRAIN_LR:-1e-2}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-1e-4}"
SEED="${SEED:-42}"
PROBE_DEVICE="${PROBE_DEVICE:-cuda:0}"
STANDARDIZE="${STANDARDIZE:-1}"
CLASS_WEIGHT="${CLASS_WEIGHT:-balanced}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
PLOT_CURVE="${PLOT_CURVE:-1}"

cmd=(
  "$PYTHON_BIN" "$ROOT_DIR/scripts/probe_utility.py"
  --feature-root "$FEATURE_ROOT"
  --feature-key "$FEATURE_KEY"
  --output "$OUTPUT_PROBE_ROOT"
  --dataset-name "$DATASET_NAME"
  --method "$METHOD"
  --stage "$STAGE"
  --label-key "$LABEL_KEY"
  --model-dir "$MODEL_DIR"
  --epochs "$TRAIN_EPOCHS"
  --batch-size "$TRAIN_BATCH_SIZE"
  --lr "$TRAIN_LR"
  --weight-decay "$TRAIN_WEIGHT_DECAY"
  --seed "$SEED"
  --device "$PROBE_DEVICE"
  --class-weight "$CLASS_WEIGHT"
)

if [[ "$FORCE_RETRAIN" == "1" ]]; then
  cmd+=(--force-retrain)
fi
if [[ "$STANDARDIZE" == "1" ]]; then
  cmd+=(--standardize)
fi
if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra=( $EXTRA_ARGS )
  cmd+=("${extra[@]}")
fi

"${cmd[@]}"

if [[ "$PLOT_CURVE" == "1" ]]; then
  set +e
  curve_plot_output="$($PYTHON_BIN "$ROOT_DIR/scripts/plot_probe_training_curve.py" --probe-root "$OUTPUT_PROBE_ROOT" 2>&1)"
  curve_plot_status=$?
  set -e
  if [[ $curve_plot_status -eq 0 ]]; then
    echo "curve_plot:       $curve_plot_output"
  elif [[ $curve_plot_status -eq 3 ]]; then
    echo "curve_plot:       skipped ($curve_plot_output)"
  else
    echo "curve_plot:       failed ($curve_plot_output)" >&2
  fi
else
  echo "curve_plot:       disabled"
fi


echo
echo "done"
echo "feature_root:   $FEATURE_ROOT"
echo "probe_root:     $OUTPUT_PROBE_ROOT"

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

DEFAULT_PAIR_FILE="$ROOT_DIR/data/lfw-dataset/lfw-deepfunneled/pairs.csv"
PAIR_FILE="${PAIR_FILE:-$DEFAULT_PAIR_FILE}"

FEATURE_KEY="${FEATURE_KEY:-hpool_priv__head_mean}"
VERIFICATION_METRIC="${VERIFICATION_METRIC:-linear_probe}"
FAR_TARGET="${FAR_TARGET:-0.01}"
PLOT_CURVE="${PLOT_CURVE:-1}"
FEATURE_KEY_DIR="${FEATURE_KEY//\//_}"
METRIC_DIR="${VERIFICATION_METRIC//\//_}"
OUTPUT_PROBE_ROOT="${OUTPUT_PROBE_ROOT:-$ROOT_DIR/runs/probes/$DATASET_NAME/$FEATURE_VERSION/privacy/$FEATURE_KEY_DIR/$METRIC_DIR}"
WORK_ROOT="${WORK_ROOT:-$ROOT_DIR/runs/probe_work/$DATASET_NAME/$FEATURE_VERSION}"
mkdir -p "$OUTPUT_PROBE_ROOT" "$WORK_ROOT"

if [[ ! -f "$PAIR_FILE" ]]; then
  echo "pair file not found: $PAIR_FILE" >&2
  exit 1
fi
if [[ ! -f "$FEATURE_ROOT/index.csv" ]]; then
  echo "missing feature index: $FEATURE_ROOT/index.csv" >&2
  exit 1
fi

pair_header="$(head -n 1 "$PAIR_FILE" | tr -d '\r')"
NORMALIZED_PAIR_FILE="$WORK_ROOT/pairs_normalized.csv"
if [[ "$pair_header" == *"sample_id_a"* && "$pair_header" == *"sample_id_b"* ]]; then
  NORMALIZED_PAIR_FILE="$PAIR_FILE"
else
  "$PYTHON_BIN" "$ROOT_DIR/scripts/convert_lfw_pairs_to_sample_ids.py" \
    --pair-file "$PAIR_FILE" \
    --index-csv "$FEATURE_ROOT/index.csv" \
    --output "$NORMALIZED_PAIR_FILE"
fi

TRAIN_EPOCHS="${TRAIN_EPOCHS:-120}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
TRAIN_LR="${TRAIN_LR:-1e-2}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-1e-4}"
SEED="${SEED:-42}"
PROBE_DEVICE="${PROBE_DEVICE:-cuda:0}"
STANDARDIZE="${STANDARDIZE:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cmd=(
  "$PYTHON_BIN" "$ROOT_DIR/scripts/probe_privacy.py"
  --feature-root "$FEATURE_ROOT"
  --feature-key "$FEATURE_KEY"
  --pair-file "$NORMALIZED_PAIR_FILE"
  --metric "$VERIFICATION_METRIC"
  --far-target "$FAR_TARGET"
  --output "$OUTPUT_PROBE_ROOT"
  --epochs "$TRAIN_EPOCHS"
  --batch-size "$TRAIN_BATCH_SIZE"
  --lr "$TRAIN_LR"
  --weight-decay "$TRAIN_WEIGHT_DECAY"
  --seed "$SEED"
  --device "$PROBE_DEVICE"
)

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
echo "feature_root:      $FEATURE_ROOT"
echo "feature_key:       $FEATURE_KEY"
echo "metric:            $VERIFICATION_METRIC"
echo "pair_file:         $PAIR_FILE"
echo "normalized_pairs:  $NORMALIZED_PAIR_FILE"
echo "probe_root:        $OUTPUT_PROBE_ROOT"

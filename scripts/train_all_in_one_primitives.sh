#!/usr/bin/env bash
# Strategy A (primitives only) — one training pass on auto-discovered PRIMITIVE folders
# (catalog families: detail / tone / exposure / depth). In all-in-one mode the trainer
# would normally include non-catalog folders too (downstream_*), so this launcher
# enumerates the catalog folders via a shell glob and passes them via TRAIN_DATASET_IDS
# as an explicit allow-list. Image loss only; no BCE gate loss. Use the resulting
# checkpoint as INIT_FROM_CHECKPOINT for Stage 3 finetune on a single downstream task.
# See docs/architecture/training_strategy.md.
source "$(dirname "$0")/_train_common.sh"

# Auto-include only primitive (catalog-family) folders unless the user pinned the pool.
if [[ -z "${TRAIN_DATASET_IDS}" ]]; then
  if [[ -d data ]]; then
    DISCOVERED=$(cd data && ls -d */ 2>/dev/null | sed 's|/$||' \
      | grep -E '^(detail|tone|exposure|depth)_.+__' || true)
    TRAIN_DATASET_IDS=$(echo "${DISCOVERED}" | tr '\n' ' ' | sed 's/[[:space:]]*$//')
  fi
fi
if [[ -z "${TRAIN_DATASET_IDS}" ]]; then
  echo "ERROR: no primitive contract folders discovered under data/; set TRAIN_DATASET_IDS explicitly." >&2
  exit 1
fi
# Default validation pool mirrors the training pool unless the user overrides it.
# VALIDATION_DATASET_IDS="${VALIDATION_DATASET_IDS:-${TRAIN_DATASET_IDS}}"

# Default validation pool covers every contract-compliant folder under data/ (anything
# with a val/ split). Lets validation also exercise downstream_* folders that are not in
# the primitive training pool. Override VALIDATION_DATASET_IDS explicitly to pin a subset.
if [[ -z "${VALIDATION_DATASET_IDS}" ]]; then
  if [[ -d data ]]; then
    DISCOVERED_VAL=$(cd data && for d in */; do
      d="${d%/}"
      [[ -d "${d}/val" ]] && echo "${d}"
    done 2>/dev/null || true)
    VALIDATION_DATASET_IDS=$(echo "${DISCOVERED_VAL}" | tr '\n' ' ' | sed 's/[[:space:]]*$//')
  fi
fi

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
VALIDATION_STEPS="${VALIDATION_STEPS:-10000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-10000}"
COMPHOSER_VALIDATION_MODE="${COMPHOSER_VALIDATION_MODE:-batch}"

RUN_TAG="${RUN_TAG:-all_in_one_primitives}"
OUTPUT_DIR="${OUTPUT_DIR:-./runs/${RUN_TAG}}"

build_common_trainer_args
TRAINER=(
  "${COMMON_TRAINER[@]}"
  --output_dir="${OUTPUT_DIR}"
  --training_strategy all_in_one
  --comphoser_validation_mode "${COMPHOSER_VALIDATION_MODE}"
  --max_train_steps="${MAX_TRAIN_STEPS}"
  --validation_steps="${VALIDATION_STEPS}"
  --checkpointing_steps="${CHECKPOINTING_STEPS}"
)
launch_trainer "Strategy A (primitives only) — all-in-one on auto-discovered primitive folders" "${TRAINER[@]}"

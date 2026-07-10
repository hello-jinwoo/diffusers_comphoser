#!/usr/bin/env bash
# Strategy A (full) — a train_A.sh spin-off that trains on BOTH primitive and downstream
# tasks together in one pass (ablation: jointly learn primitives + downstream instead of
# the pretrain-then-finetune split). The trainer auto-includes every contract folder under
# data/ when --training_strategy is all_in_one (the family catalog is not utilized on the
# supervision side; image loss only; no BCE gate loss). Override TRAIN_DATASET_IDS /
# EXCLUDE_DATASET_IDS to pin or trim the pool. See docs/architecture/training_strategy.md.
source "$(dirname "$0")/_train_common.sh"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
VALIDATION_STEPS="${VALIDATION_STEPS:-10000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-10000}"
COMPHOSER_VALIDATION_MODE="${COMPHOSER_VALIDATION_MODE:-batch}"

RUN_TAG="${RUN_TAG:-A_full}"
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
launch_trainer "Strategy A (full) — all-in-one on every contract folder under data/" "${TRAINER[@]}"

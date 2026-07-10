#!/usr/bin/env bash
# Strategy B — multi-task primitive learning with BCE gate loss.
# Sampler: group-balanced × dataset-uniform-within-group across the 4 primitive families.
# Optionally warm-start LoRA + Q-Former from train_identity.sh by passing its OUTPUT_DIR via
# INIT_FROM_CHECKPOINT. Use the resulting checkpoint as INIT_FROM_CHECKPOINT for
# train_downstream.sh to finetune on downstream tasks. See docs/architecture/training_strategy.md.
source "$(dirname "$0")/_train_common.sh"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-50000}"
VALIDATION_STEPS="${VALIDATION_STEPS:-5000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-5000}"
COMPHOSER_VALIDATION_MODE="${COMPHOSER_VALIDATION_MODE:-batch}"

RUN_TAG="${RUN_TAG:-B_primitives}"
OUTPUT_DIR="${OUTPUT_DIR:-./runs/${RUN_TAG}}"

if [[ -z "${INIT_FROM_CHECKPOINT}" ]]; then
  echo "INFO: INIT_FROM_CHECKPOINT not set; Strategy B will start from random init (no identity warmup from train_identity.sh)." >&2
fi

build_common_trainer_args
TRAINER=(
  "${COMMON_TRAINER[@]}"
  --output_dir="${OUTPUT_DIR}"
  --training_strategy step_by_step_stage2
  --comphoser_validation_mode "${COMPHOSER_VALIDATION_MODE}"
  --max_train_steps="${MAX_TRAIN_STEPS}"
  --validation_steps="${VALIDATION_STEPS}"
  --checkpointing_steps="${CHECKPOINTING_STEPS}"
)
launch_trainer "Strategy B — multi-task primitive learning (BCE gate loss)" "${TRAINER[@]}"

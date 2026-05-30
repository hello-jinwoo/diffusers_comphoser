#!/usr/bin/env bash
# Strategy B Stage 3 — downstream finetune via an additive `downstream` LoRA adapter.
# Stage 1+2 LoRA + Q-Former are frozen; only the new adapter trains. No BCE gate loss.
# Required:
#   DOWNSTREAM_TARGET_DATASET_ID — folder name under data/ for the single target task.
# Pass the Stage 2 (or any pretrained) output dir via INIT_FROM_CHECKPOINT to warm-start.
# See docs/architecture/training_strategy.md.
source "$(dirname "$0")/_train_common.sh"

DOWNSTREAM_TARGET_DATASET_ID="${DOWNSTREAM_TARGET_DATASET_ID:-}"
if [[ -z "${DOWNSTREAM_TARGET_DATASET_ID}" ]]; then
  echo "ERROR: Stage 3 requires DOWNSTREAM_TARGET_DATASET_ID (folder name under data/)." >&2
  exit 1
fi

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20000}"
VALIDATION_STEPS="${VALIDATION_STEPS:-5000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-5000}"
COMPHOSER_VALIDATION_MODE="${COMPHOSER_VALIDATION_MODE:-batch}"

RUN_TAG="${RUN_TAG:-stage3_${DOWNSTREAM_TARGET_DATASET_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-./runs/${RUN_TAG}}"

if [[ -z "${INIT_FROM_CHECKPOINT}" ]]; then
  echo "WARNING: INIT_FROM_CHECKPOINT not set; Stage 3 will start from random init (no warm-start from pretrain)." >&2
fi

build_common_trainer_args
TRAINER=(
  "${COMMON_TRAINER[@]}"
  --output_dir="${OUTPUT_DIR}"
  --training_strategy step_by_step_stage3
  --comphoser_validation_mode "${COMPHOSER_VALIDATION_MODE}"
  --max_train_steps="${MAX_TRAIN_STEPS}"
  --validation_steps="${VALIDATION_STEPS}"
  --checkpointing_steps="${CHECKPOINTING_STEPS}"
  --downstream_target_dataset_id "${DOWNSTREAM_TARGET_DATASET_ID}"
)
launch_trainer "Strategy B Stage 3 — downstream finetune on '${DOWNSTREAM_TARGET_DATASET_ID}'" "${TRAINER[@]}"

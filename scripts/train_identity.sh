#!/usr/bin/env bash
# Identity-preservation warmup — strategy-independent.
# IdentityWrapper rewrites input = target on the fly; image loss alone trains the LoRA +
# Q-Former (gate loss skipped). This is an OPTIONAL warmup you can run before either main
# training strategy: point the next launcher's INIT_FROM_CHECKPOINT at this run's OUTPUT_DIR
# to warm-start train_A.sh / train_A_full.sh (Strategy A) or train_B.sh (Strategy B).
# Default validation mode is `off` — the per-task fan-out doesn't fit identity training.
# See docs/architecture/training_strategy.md.
source "$(dirname "$0")/_train_common.sh"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
VALIDATION_STEPS="${VALIDATION_STEPS:-5000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-5000}"
COMPHOSER_VALIDATION_MODE="${COMPHOSER_VALIDATION_MODE:-off}"

RUN_TAG="${RUN_TAG:-identity}"
OUTPUT_DIR="${OUTPUT_DIR:-./runs/${RUN_TAG}}"

build_common_trainer_args
TRAINER=(
  "${COMMON_TRAINER[@]}"
  --output_dir="${OUTPUT_DIR}"
  --training_strategy step_by_step_stage1
  --comphoser_validation_mode "${COMPHOSER_VALIDATION_MODE}"
  --max_train_steps="${MAX_TRAIN_STEPS}"
  --validation_steps="${VALIDATION_STEPS}"
  --checkpointing_steps="${CHECKPOINTING_STEPS}"
)
launch_trainer "Identity-preservation warmup (input = target)" "${TRAINER[@]}"

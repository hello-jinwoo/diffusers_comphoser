#!/usr/bin/env bash
# Unified downstream finetune — trains a NEW additive `downstream` LoRA on top of a frozen
# pretrained LoRA + Q-Former (warm-started via INIT_FROM_CHECKPOINT, e.g. runs/A_primitives or
# runs/B_primitives). The pretrained LoRA + Q-Former are frozen; only the new adapter trains.
# No BCE gate loss.
#
# One script handles both pretrained inits (A or B — just point INIT_FROM_CHECKPOINT at it) and
# both training versions, selected via DOWNSTREAM_MODE:
#   integrated  — ONE additive LoRA jointly trained over all DOWNSTREAM_TASKS (uniform per folder),
#                 written to OUTPUT_DIR/.
#   respective  — the trainer loops in-process and trains a SEPARATE additive LoRA per task,
#                 written to OUTPUT_DIR/<dataset_id>/.
#
# Env knobs:
#   INIT_FROM_CHECKPOINT   pretrained run dir to warm-start from (REQUIRED in practice).
#   DOWNSTREAM_TASKS       space-separated dataset_ids under data/ (default: the four downstream_*).
#   DOWNSTREAM_MODE        integrated | respective (default: integrated).
#   OUTPUT_DIR / RUN_TAG   where checkpoints land.
# Plus every shared knob from scripts/_train_common.sh (LEARNING_RATE, TRAIN_BATCH_SIZE,
# NUM_PROCESSES, CUDA_VISIBLE_DEVICES, COMPHOSER_VALIDATION_MODE, ...).
# See docs/architecture/training_strategy.md.
source "$(dirname "$0")/_train_common.sh"

DOWNSTREAM_MODE="${DOWNSTREAM_MODE:-integrated}"
if [[ "${DOWNSTREAM_MODE}" != "integrated" && "${DOWNSTREAM_MODE}" != "respective" ]]; then
  echo "ERROR: DOWNSTREAM_MODE must be 'integrated' or 'respective' (got '${DOWNSTREAM_MODE}')." >&2
  exit 1
fi

# Default to every downstream_* application task currently under data/.
DOWNSTREAM_TASKS="${DOWNSTREAM_TASKS:-downstream_isp__DPED downstream_isp__ZRR downstream_isp__fivek downstream_llie__LOL-v2-Real}"
read -r -a DOWNSTREAM_TASK_ARRAY <<< "${DOWNSTREAM_TASKS}"
if [[ "${#DOWNSTREAM_TASK_ARRAY[@]}" -lt 1 ]]; then
  echo "ERROR: DOWNSTREAM_TASKS must list at least one dataset_id (folder name under data/)." >&2
  exit 1
fi

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20000}"
VALIDATION_STEPS="${VALIDATION_STEPS:-5000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-5000}"
COMPHOSER_VALIDATION_MODE="${COMPHOSER_VALIDATION_MODE:-batch}"

RUN_TAG="${RUN_TAG:-downstream_${DOWNSTREAM_MODE}}"
OUTPUT_DIR="${OUTPUT_DIR:-./runs/${RUN_TAG}}"

if [[ -z "${INIT_FROM_CHECKPOINT}" ]]; then
  echo "WARNING: INIT_FROM_CHECKPOINT not set; the downstream LoRA will start from a random base " \
       "instead of a pretrained primitive controller (runs/A_primitives or runs/B_primitives)." >&2
fi

build_common_trainer_args
TRAINER=(
  "${COMMON_TRAINER[@]}"
  --output_dir="${OUTPUT_DIR}"
  --training_strategy downstream
  --downstream_mode "${DOWNSTREAM_MODE}"
  --downstream_target_dataset_ids "${DOWNSTREAM_TASK_ARRAY[@]}"
  --comphoser_validation_mode "${COMPHOSER_VALIDATION_MODE}"
  --max_train_steps="${MAX_TRAIN_STEPS}"
  --validation_steps="${VALIDATION_STEPS}"
  --checkpointing_steps="${CHECKPOINTING_STEPS}"
)
launch_trainer \
  "Downstream finetune (${DOWNSTREAM_MODE}) on: ${DOWNSTREAM_TASKS}" \
  "${TRAINER[@]}"

#!/usr/bin/env bash
# FLUX + LoRA only (NO Q-Former), full — the lora_qformer-free counterpart of train_A_full.sh.
# comphoser_mode is forced to `lora_only`, so the fixed-bank controller is never built; only the
# diffusers LoRA adapter trains. Training covers BOTH primitive and downstream tasks together in
# one all_in_one pass (the trainer auto-includes every contract folder under data/). Validation
# also fans out over the ENTIRE dataset (every contract folder with a val/ split). Image loss
# only; no BCE gate loss (there is no gate to supervise without the Q-Former). The Q-Former env
# knobs from _train_common.sh are inert here. Override TRAIN_DATASET_IDS / EXCLUDE_DATASET_IDS to
# pin or trim the training pool. See docs/architecture/training_strategy.md.
source "$(dirname "$0")/_train_common.sh"

# This launcher's identity: LoRA without the Q-Former controller. Force it regardless of env.
COMPHOSER_MODE="lora_only"

# Validation pool covers the ENTIRE dataset: every contract-compliant folder under data/
# (anything with a val/ split). Override VALIDATION_DATASET_IDS explicitly to pin a subset.
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
VALIDATION_STEPS="${VALIDATION_STEPS:-5000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-10000}"
COMPHOSER_VALIDATION_MODE="${COMPHOSER_VALIDATION_MODE:-batch}"

RUN_TAG="${RUN_TAG:-lora_full}"
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
launch_trainer "FLUX + LoRA only (no Q-Former) — primitives + downstream; validation over the entire dataset" "${TRAINER[@]}"

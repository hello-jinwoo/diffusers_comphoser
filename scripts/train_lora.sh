#!/usr/bin/env bash
# FLUX + LoRA only (NO Q-Former), primitives only — the lora_qformer-free counterpart of
# train_A.sh. comphoser_mode is forced to `lora_only`, so the fixed-bank controller is never
# built; only the diffusers LoRA adapter trains on the ComPhoser dataset/runtime path. Training
# is restricted to the auto-discovered PRIMITIVE (catalog-family: detail / tone / exposure /
# depth) folders, while validation fans out over the ENTIRE dataset (every contract folder with
# a val/ split, including downstream_*). Image loss only; no BCE gate loss (there is no gate to
# supervise without the Q-Former). The Q-Former env knobs from _train_common.sh are inert here.
# See docs/architecture/training_strategy.md.
source "$(dirname "$0")/_train_common.sh"

# This launcher's identity: LoRA without the Q-Former controller. Force it regardless of env.
COMPHOSER_MODE="lora_only"

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

# Validation pool covers the ENTIRE dataset: every contract-compliant folder under data/
# (anything with a val/ split), so validation also exercises downstream_* folders that are not
# in the primitive training pool. Override VALIDATION_DATASET_IDS explicitly to pin a subset.
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

RUN_TAG="${RUN_TAG:-lora_primitives}"
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
launch_trainer "FLUX + LoRA only (no Q-Former) — primitives only; validation over the entire dataset" "${TRAINER[@]}"

#!/usr/bin/env bash
# Sourced helper for every scripts/train_*.sh launcher. Defines the shared env-var
# defaults and constructs the COMMON_TRAINER array each launcher extends with its
# strategy-specific flags. Not meant to be executed directly.

if [[ -n "${_TRAIN_COMMON_SOURCED:-}" ]]; then return 0; fi
_TRAIN_COMMON_SOURCED=1

set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
PRETRAINED_MODEL_NAME_OR_PATH="${PRETRAINED_MODEL_NAME_OR_PATH:-black-forest-labs/FLUX.2-klein-4B}"
REVISION="${REVISION:-}"
VARIANT="${VARIANT:-}"
CACHE_DIR="${CACHE_DIR:-}"

# ComPhoser knobs
PRIMITIVE_GROUPS="${PRIMITIVE_GROUPS:-detail tone exposure depth}"
COMPHOSER_MODE="${COMPHOSER_MODE:-lora_qformer}"
COMPHOSER_DATA_BACKEND="${COMPHOSER_DATA_BACKEND:-preprocessed}"
COMPHOSER_QFORMER_NUM_QUERIES="${COMPHOSER_QFORMER_NUM_QUERIES:-16}"
COMPHOSER_QFORMER_NUM_LAYERS="${COMPHOSER_QFORMER_NUM_LAYERS:-3}"
COMPHOSER_GATE_LOSS_WEIGHT_INITIAL="${COMPHOSER_GATE_LOSS_WEIGHT_INITIAL:-0.001}"
COMPHOSER_GATE_LOSS_WEIGHT_FINAL="${COMPHOSER_GATE_LOSS_WEIGHT_FINAL:-0.001}"
COMPHOSER_GATE_LOSS_WEIGHT_SCHEDULER="${COMPHOSER_GATE_LOSS_WEIGHT_SCHEDULER:-linear}"

# Logging / runtime
REPORT_TO="${REPORT_TO:-wandb}"
LOGGING_DIR="${LOGGING_DIR:-logs}"
RESOLUTION="${RESOLUTION:-1024}"
ASPECT_RATIO_BUCKETS="${ASPECT_RATIO_BUCKETS:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
ALLOW_TF32="${ALLOW_TF32:-1}"
OFFLOAD="${OFFLOAD:-0}"

# Optimizer
OPTIMIZER="${OPTIMIZER:-AdamW}"
USE_8BIT_ADAM="${USE_8BIT_ADAM:-0}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
SCALE_LR="${SCALE_LR:-0}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LR_NUM_CYCLES="${LR_NUM_CYCLES:-1}"
LR_POWER="${LR_POWER:-1.0}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-1000}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
ADAM_WEIGHT_DECAY="${ADAM_WEIGHT_DECAY:-1e-4}"
ADAM_EPSILON="${ADAM_EPSILON:-1e-8}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# LoRA
RANK="${RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_LAYERS="${LORA_LAYERS:-}"

# Flow-matching loss weighting
WEIGHTING_SCHEME="${WEIGHTING_SCHEME:-none}"
LOGIT_MEAN="${LOGIT_MEAN:-0.0}"
LOGIT_STD="${LOGIT_STD:-1.0}"
MODE_SCALE="${MODE_SCALE:-1.29}"

# Misc training knobs
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-512}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
NUM_VALIDATION_IMAGES="${NUM_VALIDATION_IMAGES:-8}"
NUM_VALIDATION_SEEDS_PER_IMAGE="${NUM_VALIDATION_SEEDS_PER_IMAGE:-1}"
# Denoising steps for controlled validation. Klein is guidance-distilled; 4 runs ~2x faster
# than the default 8 and only affects validation image fidelity, not trained weights. Empty
# = trainer default (8).
NUM_VALIDATION_INFERENCE_STEPS="${NUM_VALIDATION_INFERENCE_STEPS:-4}"
CHECKPOINTS_TOTAL_LIMIT="${CHECKPOINTS_TOTAL_LIMIT:-1}"
SKIP_FINAL_INFERENCE="${SKIP_FINAL_INFERENCE:-0}"
DISTRIBUTED_TIMEOUT_SECONDS="${DISTRIBUTED_TIMEOUT_SECONDS:-3600}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
SEED="${SEED:-0}"

# Task-pool / validation knobs
EXCLUDE_DATASET_IDS="${EXCLUDE_DATASET_IDS:-}"
TRAIN_DATASET_IDS="${TRAIN_DATASET_IDS:-}"
VALIDATION_DATASET_IDS="${VALIDATION_DATASET_IDS:-}"
VALIDATION_CHUNK_SIZE="${VALIDATION_CHUNK_SIZE:-}"
# Opt-in (default unset = no pinned task), matching the --pivotal_validation_dataset_id default.
# Set this to a dataset_id that exists under data/ to always validate it on every periodic call.
PIVOTAL_VALIDATION_DATASET_ID="${PIVOTAL_VALIDATION_DATASET_ID:-}"
VALIDATION_MODEL_CPU_OFFLOAD="${VALIDATION_MODEL_CPU_OFFLOAD:-auto}"

# Stage 1 prompt mix + cross-stage/in-stage resume
STAGE1_IDENTITY_PROMPT_MIX_RATIO="${STAGE1_IDENTITY_PROMPT_MIX_RATIO:-0.5}"
INIT_FROM_CHECKPOINT="${INIT_FROM_CHECKPOINT:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

read -r -a PRIMITIVE_GROUP_ARRAY <<< "${PRIMITIVE_GROUPS}"
if [[ "${#PRIMITIVE_GROUP_ARRAY[@]}" -lt 1 ]]; then
  echo "PRIMITIVE_GROUPS must contain at least one primitive group." >&2
  exit 1
fi

build_common_trainer_args() {
  # Populates the global COMMON_TRAINER array. Launchers call this *after* applying any
  # per-strategy env-var overrides, then append their strategy-specific flags.
  COMMON_TRAINER=(
    -m comphoser.cli.train
    --pretrained_model_name_or_path="${PRETRAINED_MODEL_NAME_OR_PATH}"
    --comphoser_mode "${COMPHOSER_MODE}"
    --comphoser_primitive_groups "${PRIMITIVE_GROUP_ARRAY[@]}"
    --comphoser_data_backend "${COMPHOSER_DATA_BACKEND}"
    --comphoser_qformer_num_queries "${COMPHOSER_QFORMER_NUM_QUERIES}"
    --comphoser_qformer_num_layers "${COMPHOSER_QFORMER_NUM_LAYERS}"
    --comphoser_gate_loss_weight_initial "${COMPHOSER_GATE_LOSS_WEIGHT_INITIAL}"
    --comphoser_gate_loss_weight_final "${COMPHOSER_GATE_LOSS_WEIGHT_FINAL}"
    --comphoser_gate_loss_weight_scheduler "${COMPHOSER_GATE_LOSS_WEIGHT_SCHEDULER}"
    --stage1_identity_prompt_mix_ratio "${STAGE1_IDENTITY_PROMPT_MIX_RATIO}"
    --report_to "${REPORT_TO}"
    --logging_dir "${LOGGING_DIR}"
    --resolution="${RESOLUTION}"
    --train_batch_size="${TRAIN_BATCH_SIZE}"
    --sample_batch_size="${SAMPLE_BATCH_SIZE}"
    --gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}"
    --optimizer="${OPTIMIZER}"
    --learning_rate="${LEARNING_RATE}"
    --lr_scheduler="${LR_SCHEDULER}"
    --lr_num_cycles="${LR_NUM_CYCLES}"
    --lr_power="${LR_POWER}"
    --lr_warmup_steps="${LR_WARMUP_STEPS}"
    --adam_beta1="${ADAM_BETA1}"
    --adam_beta2="${ADAM_BETA2}"
    --adam_weight_decay="${ADAM_WEIGHT_DECAY}"
    --adam_epsilon="${ADAM_EPSILON}"
    --max_grad_norm="${MAX_GRAD_NORM}"
    --rank="${RANK}"
    --lora_alpha="${LORA_ALPHA}"
    --lora_dropout="${LORA_DROPOUT}"
    --weighting_scheme="${WEIGHTING_SCHEME}"
    --logit_mean="${LOGIT_MEAN}"
    --logit_std="${LOGIT_STD}"
    --mode_scale="${MODE_SCALE}"
    --max_sequence_length="${MAX_SEQUENCE_LENGTH}"
    --guidance_scale="${GUIDANCE_SCALE}"
    --num_train_epochs="${NUM_TRAIN_EPOCHS}"
    --num_validation_images="${NUM_VALIDATION_IMAGES}"
    --num_validation_seeds_per_image="${NUM_VALIDATION_SEEDS_PER_IMAGE}"
    --checkpoints_total_limit="${CHECKPOINTS_TOTAL_LIMIT}"
    --mixed_precision="${MIXED_PRECISION}"
    --distributed_timeout_seconds "${DISTRIBUTED_TIMEOUT_SECONDS}"
    --dataloader_num_workers="${DATALOADER_NUM_WORKERS}"
    --seed="${SEED}"
  )
  if [[ -n "${REVISION}" ]]; then COMMON_TRAINER+=(--revision "${REVISION}"); fi
  if [[ -n "${VARIANT}" ]]; then COMMON_TRAINER+=(--variant "${VARIANT}"); fi
  if [[ -n "${CACHE_DIR}" ]]; then COMMON_TRAINER+=(--cache_dir "${CACHE_DIR}"); fi
  if [[ -n "${ASPECT_RATIO_BUCKETS}" ]]; then COMMON_TRAINER+=(--aspect_ratio_buckets "${ASPECT_RATIO_BUCKETS}"); fi
  if [[ -n "${LORA_LAYERS}" ]]; then COMMON_TRAINER+=(--lora_layers "${LORA_LAYERS}"); fi
  if [[ "${ENABLE_GRADIENT_CHECKPOINTING}" == "1" ]]; then COMMON_TRAINER+=(--gradient_checkpointing); fi
  if [[ "${ALLOW_TF32}" == "1" ]]; then COMMON_TRAINER+=(--allow_tf32); fi
  if [[ "${USE_8BIT_ADAM}" == "1" ]]; then COMMON_TRAINER+=(--use_8bit_adam); fi
  if [[ "${SCALE_LR}" == "1" ]]; then COMMON_TRAINER+=(--scale_lr); fi
  if [[ "${OFFLOAD}" == "1" ]]; then COMMON_TRAINER+=(--offload); fi
  if [[ "${SKIP_FINAL_INFERENCE}" == "1" ]]; then COMMON_TRAINER+=(--skip_final_inference); fi
  if [[ -n "${EXCLUDE_DATASET_IDS}" ]]; then
    read -r -a EXCLUDE_DATASET_ID_ARRAY <<< "${EXCLUDE_DATASET_IDS}"
    COMMON_TRAINER+=(--exclude_dataset_ids "${EXCLUDE_DATASET_ID_ARRAY[@]}")
  fi
  if [[ -n "${TRAIN_DATASET_IDS}" ]]; then
    read -r -a TRAIN_DATASET_ID_ARRAY <<< "${TRAIN_DATASET_IDS}"
    COMMON_TRAINER+=(--train_dataset_ids "${TRAIN_DATASET_ID_ARRAY[@]}")
  fi
  if [[ -n "${VALIDATION_DATASET_IDS}" ]]; then
    read -r -a VALIDATION_DATASET_ID_ARRAY <<< "${VALIDATION_DATASET_IDS}"
    COMMON_TRAINER+=(--validation_dataset_ids "${VALIDATION_DATASET_ID_ARRAY[@]}")
  fi
  if [[ -n "${VALIDATION_CHUNK_SIZE}" ]]; then
    COMMON_TRAINER+=(--validation_chunk_size "${VALIDATION_CHUNK_SIZE}")
  fi
  if [[ -n "${PIVOTAL_VALIDATION_DATASET_ID}" ]]; then
    COMMON_TRAINER+=(--pivotal_validation_dataset_id "${PIVOTAL_VALIDATION_DATASET_ID}")
  fi
  if [[ -n "${VALIDATION_MODEL_CPU_OFFLOAD}" ]]; then
    COMMON_TRAINER+=(--validation_model_cpu_offload "${VALIDATION_MODEL_CPU_OFFLOAD}")
  fi
  if [[ -n "${NUM_VALIDATION_INFERENCE_STEPS}" ]]; then
    COMMON_TRAINER+=(--num_validation_inference_steps "${NUM_VALIDATION_INFERENCE_STEPS}")
  fi
  if [[ -n "${INIT_FROM_CHECKPOINT}" ]]; then
    COMMON_TRAINER+=(--init_from_checkpoint "${INIT_FROM_CHECKPOINT}")
  fi
  if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    COMMON_TRAINER+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
  fi
}

launch_trainer() {
  local label="$1"
  shift
  local stage_args=("$@")
  echo "============================================================"
  echo "Launching ${label}"
  echo "  cuda_visible_devices: ${CUDA_VISIBLE_DEVICES}"
  echo "  primitive_groups: ${PRIMITIVE_GROUPS}"
  if [[ -n "${TRAIN_DATASET_IDS}" ]]; then
    echo "  train_dataset_ids: ${TRAIN_DATASET_IDS}"
  fi
  if [[ -n "${VALIDATION_DATASET_IDS}" ]]; then
    echo "  validation_dataset_ids: ${VALIDATION_DATASET_IDS}"
  fi
  echo "============================================================"
  if [[ "${NUM_PROCESSES}" == "1" ]]; then
    python "${stage_args[@]}"
  else
    accelerate launch --num_processes "${NUM_PROCESSES}" "${stage_args[@]}"
  fi
}

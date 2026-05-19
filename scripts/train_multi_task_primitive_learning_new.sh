#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
PRETRAINED_MODEL_NAME_OR_PATH="${PRETRAINED_MODEL_NAME_OR_PATH:-black-forest-labs/FLUX.2-klein-4B}"
REVISION="${REVISION:-}"
VARIANT="${VARIANT:-}"
CACHE_DIR="${CACHE_DIR:-}"

PRIMITIVE_GROUPS="${PRIMITIVE_GROUPS:-detail tone exposure depth}"
COMPHOSER_MODE="${COMPHOSER_MODE:-lora_qformer}"
COMPHOSER_DATA_BACKEND="${COMPHOSER_DATA_BACKEND:-preprocessed}"
COMPHOSER_VALIDATION_MODE="${COMPHOSER_VALIDATION_MODE:-batch}"
COMPHOSER_QFORMER_NUM_QUERIES="${COMPHOSER_QFORMER_NUM_QUERIES:-16}"
COMPHOSER_QFORMER_NUM_LAYERS="${COMPHOSER_QFORMER_NUM_LAYERS:-3}"
COMPHOSER_GATE_LOSS_WEIGHT_INITIAL="${COMPHOSER_GATE_LOSS_WEIGHT_INITIAL:-0.001}"
COMPHOSER_GATE_LOSS_WEIGHT_FINAL="${COMPHOSER_GATE_LOSS_WEIGHT_FINAL:-0.001}"
COMPHOSER_GATE_LOSS_WEIGHT_SCHEDULER="${COMPHOSER_GATE_LOSS_WEIGHT_SCHEDULER:-linear}"

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

RANK="${RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_LAYERS="${LORA_LAYERS:-}"

WEIGHTING_SCHEME="${WEIGHTING_SCHEME:-none}"
LOGIT_MEAN="${LOGIT_MEAN:-0.0}"
LOGIT_STD="${LOGIT_STD:-1.0}"
MODE_SCALE="${MODE_SCALE:-1.29}"

MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-512}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-50000}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
VALIDATION_STEPS="${VALIDATION_STEPS:-5000}"
NUM_VALIDATION_IMAGES="${NUM_VALIDATION_IMAGES:-8}"
NUM_VALIDATION_SEEDS_PER_IMAGE="${NUM_VALIDATION_SEEDS_PER_IMAGE:-2}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-5000}"
CHECKPOINTS_TOTAL_LIMIT="${CHECKPOINTS_TOTAL_LIMIT:-1}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
SKIP_FINAL_INFERENCE="${SKIP_FINAL_INFERENCE:-0}"
DISTRIBUTED_TIMEOUT_SECONDS="${DISTRIBUTED_TIMEOUT_SECONDS:-3600}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
SEED="${SEED:-0}"

RUN_TAG="${RUN_TAG:-router_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-./runs/primitives_${RUN_TAG}_${COMPHOSER_MODE}_lr_${LEARNING_RATE}_${LR_SCHEDULER}_ncycle_${LR_NUM_CYCLES}_gate_loss_${COMPHOSER_GATE_LOSS_WEIGHT_INITIAL}_${COMPHOSER_GATE_LOSS_WEIGHT_FINAL}_iter_${MAX_TRAIN_STEPS}}"

read -r -a PRIMITIVE_GROUP_ARRAY <<< "${PRIMITIVE_GROUPS}"
if [[ "${#PRIMITIVE_GROUP_ARRAY[@]}" -lt 2 ]]; then
  echo "PRIMITIVE_GROUPS must contain at least two primitive groups for the multi-task launcher." >&2
  exit 1
fi

TRAINER=(
  -m comphoser.cli.train
  --pretrained_model_name_or_path="${PRETRAINED_MODEL_NAME_OR_PATH}"
  --output_dir="${OUTPUT_DIR}"
  --comphoser_mode "${COMPHOSER_MODE}"
  --comphoser_primitive_groups "${PRIMITIVE_GROUP_ARRAY[@]}"
  --comphoser_data_backend "${COMPHOSER_DATA_BACKEND}"
  --comphoser_validation_mode "${COMPHOSER_VALIDATION_MODE}"
  --comphoser_qformer_num_queries "${COMPHOSER_QFORMER_NUM_QUERIES}"
  --comphoser_qformer_num_layers "${COMPHOSER_QFORMER_NUM_LAYERS}"
  --comphoser_gate_loss_weight_initial "${COMPHOSER_GATE_LOSS_WEIGHT_INITIAL}"
  --comphoser_gate_loss_weight_final "${COMPHOSER_GATE_LOSS_WEIGHT_FINAL}"
  --comphoser_gate_loss_weight_scheduler "${COMPHOSER_GATE_LOSS_WEIGHT_SCHEDULER}"
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
  --max_train_steps="${MAX_TRAIN_STEPS}"
  --validation_steps="${VALIDATION_STEPS}"
  --num_validation_images="${NUM_VALIDATION_IMAGES}"
  --num_validation_seeds_per_image="${NUM_VALIDATION_SEEDS_PER_IMAGE}"
  --checkpointing_steps="${CHECKPOINTING_STEPS}"
  --checkpoints_total_limit="${CHECKPOINTS_TOTAL_LIMIT}"
  --mixed_precision="${MIXED_PRECISION}"
  --distributed_timeout_seconds "${DISTRIBUTED_TIMEOUT_SECONDS}"
  --dataloader_num_workers="${DATALOADER_NUM_WORKERS}"
  --seed="${SEED}"
)

if [[ -n "${REVISION}" ]]; then
  TRAINER+=(--revision "${REVISION}")
fi
if [[ -n "${VARIANT}" ]]; then
  TRAINER+=(--variant "${VARIANT}")
fi
if [[ -n "${CACHE_DIR}" ]]; then
  TRAINER+=(--cache_dir "${CACHE_DIR}")
fi
if [[ -n "${ASPECT_RATIO_BUCKETS}" ]]; then
  TRAINER+=(--aspect_ratio_buckets "${ASPECT_RATIO_BUCKETS}")
fi
if [[ -n "${LORA_LAYERS}" ]]; then
  TRAINER+=(--lora_layers "${LORA_LAYERS}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  TRAINER+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ "${ENABLE_GRADIENT_CHECKPOINTING}" == "1" ]]; then
  TRAINER+=(--gradient_checkpointing)
fi
if [[ "${ALLOW_TF32}" == "1" ]]; then
  TRAINER+=(--allow_tf32)
fi
if [[ "${USE_8BIT_ADAM}" == "1" ]]; then
  TRAINER+=(--use_8bit_adam)
fi
if [[ "${SCALE_LR}" == "1" ]]; then
  TRAINER+=(--scale_lr)
fi
if [[ "${OFFLOAD}" == "1" ]]; then
  TRAINER+=(--offload)
fi
if [[ "${SKIP_FINAL_INFERENCE}" == "1" ]]; then
  TRAINER+=(--skip_final_inference)
fi

echo "Launching ComPhoser multi-task training"
echo "  router: prompt_router_v2 predicted_only (code default for lora_qformer)"
echo "  cuda_visible_devices: ${CUDA_VISIBLE_DEVICES}"
echo "  primitive_groups: ${PRIMITIVE_GROUPS}"
echo "  output_dir: ${OUTPUT_DIR}"

if [[ "${NUM_PROCESSES}" == "1" ]]; then
  python "${TRAINER[@]}"
else
  accelerate launch --num_processes "${NUM_PROCESSES}" "${TRAINER[@]}"
fi

#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
PRETRAINED_MODEL_NAME_OR_PATH="${PRETRAINED_MODEL_NAME_OR_PATH:-black-forest-labs/FLUX.2-klein-4B}"
PRIMITIVE_GROUP="${PRIMITIVE_GROUP:-detail}"
COMPHOSER_MODE="${COMPHOSER_MODE:-lora_qformer}"
COMPHOSER_DATA_BACKEND="${COMPHOSER_DATA_BACKEND:-preprocessed}"
COMPHOSER_VALIDATION_MODE="${COMPHOSER_VALIDATION_MODE:-batch}"
COMPHOSER_GATE_LOSS_WEIGHT="${COMPHOSER_GATE_LOSS_WEIGHT:-0.1}"
REPORT_TO="${REPORT_TO:-wandb}"
OUTPUT_DIR="${OUTPUT_DIR:-./runs/single_task_${PRIMITIVE_GROUP}_${COMPHOSER_MODE}}"
RESOLUTION="${RESOLUTION:-1024}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
OPTIMIZER="${OPTIMIZER:-AdamW}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LR_SCHEDULER="${LR_SCHEDULER:-constant}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
RANK="${RANK:-8}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"
VALIDATION_STEPS="${VALIDATION_STEPS:-50}"
NUM_VALIDATION_IMAGES="${NUM_VALIDATION_IMAGES:-2}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-100}"
CHECKPOINTS_TOTAL_LIMIT="${CHECKPOINTS_TOTAL_LIMIT:-5}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
SEED="${SEED:-0}"

TRAINER=(
  -m comphoser.cli.train
  --pretrained_model_name_or_path="${PRETRAINED_MODEL_NAME_OR_PATH}"
  --output_dir="${OUTPUT_DIR}"
  --comphoser_mode "${COMPHOSER_MODE}"
  --comphoser_primitive_groups "${PRIMITIVE_GROUP}"
  --comphoser_data_backend "${COMPHOSER_DATA_BACKEND}"
  --comphoser_validation_mode "${COMPHOSER_VALIDATION_MODE}"
  --comphoser_gate_loss_weight "${COMPHOSER_GATE_LOSS_WEIGHT}"
  --report_to "${REPORT_TO}"
  --resolution="${RESOLUTION}"
  --train_batch_size="${TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}"
  --gradient_checkpointing
  --optimizer="${OPTIMIZER}"
  --learning_rate="${LEARNING_RATE}"
  --lr_scheduler="${LR_SCHEDULER}"
  --lr_warmup_steps="${LR_WARMUP_STEPS}"
  --rank="${RANK}"
  --max_train_steps="${MAX_TRAIN_STEPS}"
  --validation_steps="${VALIDATION_STEPS}"
  --num_validation_images="${NUM_VALIDATION_IMAGES}"
  --checkpointing_steps="${CHECKPOINTING_STEPS}"
  --checkpoints_total_limit="${CHECKPOINTS_TOTAL_LIMIT}"
  --mixed_precision="${MIXED_PRECISION}"
  --dataloader_num_workers="${DATALOADER_NUM_WORKERS}"
  --seed="${SEED}"
)

if [[ "${NUM_PROCESSES}" == "1" ]]; then
  python "${TRAINER[@]}"
else
  accelerate launch --num_processes "${NUM_PROCESSES}" "${TRAINER[@]}"
fi

#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
PRETRAINED_MODEL_NAME_OR_PATH="${PRETRAINED_MODEL_NAME_OR_PATH:-black-forest-labs/FLUX.2-klein-4B}"
PRIMITIVE_GROUPS="${PRIMITIVE_GROUPS:-detail tone exposure depth}"
COMPHOSER_MODE="${COMPHOSER_MODE:-lora_qformer}"
# COMPHOSER_MODE="${COMPHOSER_MODE:-lora_only}"
COMPHOSER_DATA_BACKEND="${COMPHOSER_DATA_BACKEND:-preprocessed}"
COMPHOSER_VALIDATION_MODE="${COMPHOSER_VALIDATION_MODE:-batch}"
COMPHOSER_GATE_LOSS_WEIGHT_INITIAL="${COMPHOSER_GATE_LOSS_WEIGHT_INITIAL:-0.003}"
COMPHOSER_GATE_LOSS_WEIGHT_FINAL="${COMPHOSER_GATE_LOSS_WEIGHT_FINAL:-0.003}"
COMPHOSER_GATE_LOSS_WEIGHT_SCHEDULER="${COMPHOSER_GATE_LOSS_WEIGHT_SCHEDULER:-linear}"
REPORT_TO="${REPORT_TO:-wandb}"
RESOLUTION="${RESOLUTION:-1024}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
OPTIMIZER="${OPTIMIZER:-AdamW}"
LEARNING_RATE="${LEARNING_RATE:-4e-4}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LR_NUM_CYCLES="${LR_NUM_CYCLES:-1}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-1000}"
RANK="${RANK:-16}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-50000}"
VALIDATION_STEPS="${VALIDATION_STEPS:-5000}"
NUM_VALIDATION_IMAGES="${NUM_VALIDATION_IMAGES:-8}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-5000}"
CHECKPOINTS_TOTAL_LIMIT="${CHECKPOINTS_TOTAL_LIMIT:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DISTRIBUTED_TIMEOUT_SECONDS="${DISTRIBUTED_TIMEOUT_SECONDS:-3600}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-./runs/multi_task_primitives_${COMPHOSER_MODE}_lr_${LEARNING_RATE}_${LR_SCHEDULER}_ncycle_${LR_NUM_CYCLES}_gate_loss_${COMPHOSER_GATE_LOSS_WEIGHT_INITIAL}_${COMPHOSER_GATE_LOSS_WEIGHT_FINAL}_scheduler_${COMPHOSER_GATE_LOSS_WEIGHT_SCHEDULER}_iter_${MAX_TRAIN_STEPS}}"

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
  --comphoser_gate_loss_weight_initial "${COMPHOSER_GATE_LOSS_WEIGHT_INITIAL}"
  --comphoser_gate_loss_weight_final "${COMPHOSER_GATE_LOSS_WEIGHT_FINAL}"
  --comphoser_gate_loss_weight_scheduler "${COMPHOSER_GATE_LOSS_WEIGHT_SCHEDULER}"
  --report_to "${REPORT_TO}"
  --resolution="${RESOLUTION}"
  --train_batch_size="${TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}"
  --gradient_checkpointing
  --optimizer="${OPTIMIZER}"
  --learning_rate="${LEARNING_RATE}"
  --lr_scheduler="${LR_SCHEDULER}"
  --lr_num_cycles="${LR_NUM_CYCLES}"
  --lr_warmup_steps="${LR_WARMUP_STEPS}"
  --rank="${RANK}"
  --max_train_steps="${MAX_TRAIN_STEPS}"
  --validation_steps="${VALIDATION_STEPS}"
  --num_validation_images="${NUM_VALIDATION_IMAGES}"
  --checkpointing_steps="${CHECKPOINTING_STEPS}"
  --checkpoints_total_limit="${CHECKPOINTS_TOTAL_LIMIT}"
  --mixed_precision="${MIXED_PRECISION}"
  --distributed_timeout_seconds "${DISTRIBUTED_TIMEOUT_SECONDS}"
  --dataloader_num_workers="${DATALOADER_NUM_WORKERS}"
  --seed="${SEED}"
)

if [[ "${NUM_PROCESSES}" == "1" ]]; then
  python "${TRAINER[@]}"
else
  accelerate launch --num_processes "${NUM_PROCESSES}" "${TRAINER[@]}"
fi

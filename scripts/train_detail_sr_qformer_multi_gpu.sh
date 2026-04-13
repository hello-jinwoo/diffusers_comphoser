#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

NUM_PROCESSES="${NUM_PROCESSES:-2}"

accelerate launch --num_processes "${NUM_PROCESSES}" \
  examples/dreambooth/train_dreambooth_lora_flux2_klein_img2img.py \
  --pretrained_model_name_or_path="${PRETRAINED_MODEL_NAME_OR_PATH:-black-forest-labs/FLUX.2-klein-4B}" \
  --output_dir="${OUTPUT_DIR:-./runs/detail_sr_real_lora_qformer_2gpu}" \
  --comphoser_mode lora_qformer \
  --comphoser_primitive_groups detail \
  --comphoser_data_backend preprocessed \
  --comphoser_validation_mode batch \
  --resolution="${RESOLUTION:-1024}" \
  --train_batch_size="${TRAIN_BATCH_SIZE:-4}" \
  --gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --gradient_checkpointing \
  --optimizer="${OPTIMIZER:-AdamW}" \
  --learning_rate="${LEARNING_RATE:-1e-4}" \
  --lr_scheduler="${LR_SCHEDULER:-constant}" \
  --lr_warmup_steps="${LR_WARMUP_STEPS:-0}" \
  --rank="${RANK:-8}" \
  --max_train_steps="${MAX_TRAIN_STEPS:-1000}" \
  --validation_steps="${VALIDATION_STEPS:-10}" \
  --num_validation_images="${NUM_VALIDATION_IMAGES:-2}" \
  --checkpointing_steps="${CHECKPOINTING_STEPS:-10}" \
  --checkpoints_total_limit="${CHECKPOINTS_TOTAL_LIMIT:-3}" \
  --mixed_precision="${MIXED_PRECISION:-bf16}" \
  --dataloader_num_workers="${DATALOADER_NUM_WORKERS:-0}" \
  --seed="${SEED:-0}"

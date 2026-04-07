# FLUX.2-Klein LoRA Img2Img Trainer

This repository is a trimmed copy of `diffusers` kept for one workflow only:
LoRA training of FLUX.2-Klein in an image-to-image setting.

What remains:

- core library code in `src/diffusers`
- the trainer at `examples/dreambooth/train_dreambooth_lora_flux2_klein_img2img.py`
- the dependencies file at `examples/dreambooth/requirements_flux.txt`

## Setup

FLUX.2-Klein is gated on Hugging Face. Accept the model license first, then log in:

```bash
hf auth login
```

Install the library and trainer dependencies:

```bash
pip install -e .
pip install -r examples/dreambooth/requirements_flux.txt
accelerate config default
```

Optional packages:

- `bitsandbytes` for `--use_8bit_adam` or `--bnb_quantization_config_path`
- `torchao` for `--do_fp8_training`
- `prodigyopt` for `--optimizer=prodigy`
- `wandb` for `--report_to=wandb`

## Dataset Format

For paired img2img training, use `--dataset_name` with either:

- a Hugging Face dataset id
- a local datasets-compatible directory

Your dataset should expose three columns:

- target image
- conditioning image
- instruction or caption

Pass those column names with:

- `--image_column`
- `--cond_image_column`
- `--caption_column`

## Example

Download a tiny paired local dataset first:

```bash
python scripts/download_flux2_klein_lora_smoke_data.py --overwrite
```

This writes a local `imagefolder` dataset under `./sample_data/flux2_klein_lora_smoke` with:

- target column: `image`
- conditioning-image column: `conditioning_image`
- prompt column: `instruction`

Single-GPU end-to-end command with final inference:

```bash
accelerate launch --num_processes=1 examples/dreambooth/train_dreambooth_lora_flux2_klein_img2img.py \
  --pretrained_model_name_or_path=black-forest-labs/FLUX.2-klein-4B \
  --dataset_name=./sample_data/flux2_klein_lora_smoke \
  --image_column=image \
  --cond_image_column=conditioning_image \
  --caption_column=instruction \
  --output_dir=./runs/flux2-klein-img2img-lora-end2end-single-gpu \
  --cache_latents \
  --gradient_checkpointing \
  --mixed_precision=bf16 \
  --resolution=512 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=1 \
  --optimizer=AdamW \
  --learning_rate=1e-4 \
  --lr_scheduler=constant \
  --lr_warmup_steps=0 \
  --max_train_steps=10 \
  --rank=8 \
  --seed=0 \
  --dataloader_num_workers=0 \
  --validation_prompt="restore the butterfly photo from the blurred low-resolution reference while preserving natural wing patterns and color" \
  --validation_image=./sample_data/flux2_klein_lora_smoke/train/conditioning/000.png \
  --num_validation_images=1 \
  --validation_epochs=10
```

Multi-GPU end-to-end command with final inference:

```bash
accelerate launch --multi_gpu --num_processes=2 examples/dreambooth/train_dreambooth_lora_flux2_klein_img2img.py \
  --pretrained_model_name_or_path=black-forest-labs/FLUX.2-klein-4B \
  --dataset_name=./sample_data/flux2_klein_lora_smoke \
  --image_column=image \
  --cond_image_column=conditioning_image \
  --caption_column=instruction \
  --output_dir=./runs/flux2-klein-img2img-lora-end2end-multi-gpu \
  --cache_latents \
  --gradient_checkpointing \
  --mixed_precision=bf16 \
  --resolution=512 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=1 \
  --optimizer=AdamW \
  --learning_rate=1e-4 \
  --lr_scheduler=constant \
  --lr_warmup_steps=0 \
  --max_train_steps=10 \
  --rank=8 \
  --seed=0 \
  --dataloader_num_workers=0 \
  --validation_prompt="restore the butterfly photo from the blurred low-resolution reference while preserving natural wing patterns and color" \
  --validation_image=./sample_data/flux2_klein_lora_smoke/train/conditioning/000.png \
  --num_validation_images=1 \
  --validation_epochs=10
```

The distributed synchronization fix in the trainer keeps the other ranks waiting correctly while rank 0 runs validation or final inference, so this end-to-end variant should finish cleanly instead of hanging at the end.

Use `black-forest-labs/FLUX.2-klein-9B` instead if you want the 9B variant.

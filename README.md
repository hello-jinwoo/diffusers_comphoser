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

```bash
accelerate launch examples/dreambooth/train_dreambooth_lora_flux2_klein_img2img.py \
  --pretrained_model_name_or_path=black-forest-labs/FLUX.2-klein-4B \
  --dataset_name=/path/to/dataset \
  --image_column=output \
  --cond_image_column=file_name \
  --caption_column=instruction \
  --output_dir=./flux2-klein-img2img-lora \
  --cache_latents \
  --gradient_checkpointing \
  --resolution=1024 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --optimizer=adamw \
  --learning_rate=1e-4 \
  --lr_scheduler=constant \
  --lr_warmup_steps=100 \
  --max_train_steps=1000 \
  --rank=16 \
  --seed=0
```

Optional validation:

```bash
  --validation_prompt="your prompt" \
  --validation_image=/path/to/conditioning-image.png \
  --num_validation_images=1
```

Use `black-forest-labs/FLUX.2-klein-9B` instead if you want the 9B variant.

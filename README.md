# ComPhoser

## 1. Overview

ComPhoser is a diffusion-based computational photography project built on top of FLUX.2 Klein. The current repository keeps the working FLUX.2 Klein image-to-image LoRA baseline and extends it with the first ComPhoser Q-Former conditioning path for a single primitive task.

The current implemented pilot is `detail_sr_x4`, mapped from `primitive_groups=["detail"]` and backed by the rebuilt SR dataset at `data/detail_sr__RealSR_v3/`.

## 2. Project Status

- [x] FLUX.2 Klein image-to-image LoRA baseline retained in this repository
- [x] Single primitive SR pilot for `detail_sr_x4`
- [x] Q-Former-enabled pilot training path
- [x] Controlled validation export under `output_dir/comphoser/controlled_validation/`
- [ ] Multiple primitive training
- [ ] Downstream task training on top of primitives

Notes:

- The first real `lora_only` SR run completed on `2026-04-10`.
- The first real `lora_qformer` SR run completed on `2026-04-10`.
- Periodic multi-GPU validation now rebuilds detached validation models before offload so rank `0` validation does not mutate the live training transformer or Q-Former modules.
- The main open issue is controller behavior quality, not basic training/export wiring.

## 3. Settings

### Requirements

Accept the gated FLUX.2 Klein license on Hugging Face, then log in:

```bash
hf auth login
```

Install the repository and training dependencies:

```bash
pip install -e .
pip install -r examples/dreambooth/requirements_flux.txt
accelerate config default
```

Optional packages:

- `bitsandbytes` for `--use_8bit_adam` or quantized loading
- `torchao` for FP8 training paths
- `prodigyopt` for `--optimizer=prodigy`
- `wandb` for `--report_to=wandb`

### Dataset

The active single-primitive dataset is:

- `data/detail_sr__RealSR_v3/`

Current validated dataset state:

- `train`: 400 paired samples
- `val`: 100 paired samples
- both splits contain `original/`, `raw/`, and `preprocessed/`

Dataset naming follows:

```text
data/{group}_{task}__{dataset_name}/
```

Expected split structure:

```text
data/
  {dataname}/
    train/
      raw/
        images/
          input/
          target/
        prompt/
      preprocessed/
        image_latent_cache/
          input/
          target/
        prompt_latent_cache/
    val/
      (same as train/)
```

Dataset preparation flow:

1. Place relocated source assets under `data/{dataname}/{train_or_val}/original/`.
2. Build resized and renamed raw pairs.
3. Create prompt text files under `raw/prompt/`.
4. Build split-local latent and prompt caches under `preprocessed/`.

Builders:

```bash
PYTHONPATH=src python scripts/build_comphoser_raw_dataset.py \
  --dataset_root data/{dataname}/{train_or_val} \
  --pairing_mode by_name
```

```bash
PYTHONPATH=src python scripts/build_comphoser_preprocessed_dataset.py \
  --dataset_root data/{dataname}/{train_or_val} \
  --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-4B
```

For ComPhoser pilot modes, the default backend is `--comphoser_data_backend preprocessed`.

## 4. Training

ComPhoser currently exposes three training directions.

### [1] Single primitive

Status: completed for SR (`detail_sr_x4`)

Concrete training script:

- `examples/dreambooth/train_dreambooth_lora_flux2_klein_img2img.py`

Run scripts:
```bash
bash scripts/train_detail_sr_qformer_single_gpu.sh
```

```bash
bash scripts/train_detail_sr_qformer_multi_gpu.sh
```

The scripts expose the main knobs through environment variables such as `OUTPUT_DIR`, `CUDA_VISIBLE_DEVICES`, and `NUM_PROCESSES`.

Supported pilot modes:

- `baseline`: retained FLUX.2 Klein LoRA img2img path
- `lora_only`: ComPhoser dataset routing without Q-Former conditioning
- `lora_qformer`: ComPhoser dataset routing with the Q-Former enabled

### [2] Multiple primitive

Status: TODO

### [3] Downstream tasks

Status: TODO

## 5. Inference

The current supported inference surface is the ComPhoser controlled-validation path, not a separate polished standalone inference CLI.

For the completed SR primitive, `lora_qformer` final export writes:

- `output_dir/comphoser/metadata.json`
- `output_dir/comphoser/shared_qformer.safetensors`
- `output_dir/comphoser/task_query_bank.safetensors`
- `output_dir/comphoser/controlled_validation/summary.json`
- `output_dir/comphoser/controlled_validation/images/`

`batch` validation is the default inference-style export for the SR pilot. It runs only the active mode for the current training run:

- `baseline -> flux_only`
- `lora_only -> lora_only`
- `lora_qformer -> lora_qformer`

For each selected validation sample it writes:

- `{image_id}_input.png`
- `{image_id}_output_1.png`, `{image_id}_output_2.png`, ...
- `{image_id}_gt.png`
- `{image_id}_all.png`

`--num_validation_images` controls how many validation input samples are processed in ComPhoser batch validation, and `--num_validation_seeds_per_image` controls how many seeded outputs are generated per sample.

Optional single-case validation remains available:

```bash
PYTHONPATH=src python -m comphoser.cli.train \
  --pretrained_model_name_or_path=black-forest-labs/FLUX.2-klein-4B \
  --output_dir=./runs/detail_sr_single_validation \
  --comphoser_mode lora_qformer \
  --comphoser_primitive_groups detail \
  --comphoser_validation_mode single \
  --validation_prompt "restore fine detail conservatively" \
  --validation_image ./some_validation_input.png \
  --num_validation_seeds_per_image 2
```

Current limitation:

- multi-primitive composed inference is not implemented yet

# ComPhoser

ComPhoser is a FLUX.2 Klein image-to-image training repo with a ComPhoser Q-Former control path for computational photography primitives.

Current supported primitive groups:

- `detail`
- `tone`
- `exposure`
- `depth`

If you only need to get started, follow the sections in this order:

1. `Setup`
2. `Prepare A Dataset`
3. `Train`
4. `Check Outputs`
5. `Evaluate A Checkpoint`

## Setup

Accept the gated FLUX.2 Klein license on Hugging Face, then authenticate:

```bash
hf auth login
```

Install the repo and training dependencies:

```bash
pip install -e .
pip install -r examples/dreambooth/requirements_flux.txt
accelerate config default
```

Optional packages:

- `bitsandbytes` for 8-bit optimizer or quantized loading
- `torchao` for FP8 paths
- `prodigyopt` for `--optimizer=prodigy`
- `wandb` for experiment logging

## What You Can Run Today

Training modes:

- `baseline`: retained FLUX.2 Klein LoRA img2img path
- `lora_only`: ComPhoser dataset/runtime path without Q-Former conditioning
- `lora_qformer`: ComPhoser dataset/runtime path with the fixed-bank Q-Former

Canonical primitive groups:

- `detail`
- `tone`
- `exposure`
- `depth`

Notes:

- Use `tone`, not `tone_color`. `tone_color` is only a compatibility alias.
- The current controller is a fixed-bank v1 design: `4` families x `4` query slots = `16` total query tokens.
- Current training is dataset-backed for all four groups, but supervision still assumes at most one active primitive family per sample.
- Controlled-validation summaries use schema `comphoser-controlled-validation-v6` and report PSNR, SSIM, mean CIEDE2000 Delta E, and Q-Former gate diagnostics when available.

## Prepare A Dataset

ComPhoser datasets follow this naming pattern:

```text
data/{group}_{task}__{dataset_name}/
```

Expected layout per split:

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
      raw/
      preprocessed/
```

Recommended flow:

1. Create the contract-aligned `raw/` pairs and prompt files.
2. Build the `preprocessed/` latent caches.
3. Use `--comphoser_data_backend preprocessed` for training.

Build `raw/` for a split:

```bash
PYTHONPATH=src python scripts/build_comphoser_raw_dataset.py \
  --dataset_root data/{dataname}/{train_or_val} \
  --pairing_mode by_name
```

Build `preprocessed/` for a split:

```bash
PYTHONPATH=src python scripts/build_comphoser_preprocessed_dataset.py \
  --dataset_root data/{dataname}/{train_or_val} \
  --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-4B
```

Current validated dataset-backed roots in this checkout:

- `data/detail_sr__RealSR_v3/`
- `data/tone_style__FilmSet/`
- `data/exposure_ec__MSEC/`
- `data/depth_bokeh__RealBokeh/`

## Train

### Fastest Path

Single primitive:

```bash
bash scripts/train_single_task_primitive_learning.sh
```

Multiple primitives:

```bash
bash scripts/train_multi_task_primitive_learning.sh
```

### Common Examples

Single-task `detail` run on one GPU:

```bash
PRIMITIVE_GROUP=detail \
NUM_PROCESSES=1 \
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_DIR=./runs/single_task_detail \
bash scripts/train_single_task_primitive_learning.sh
```

Single-task `tone` run on two GPUs:

```bash
PRIMITIVE_GROUP=tone \
NUM_PROCESSES=2 \
CUDA_VISIBLE_DEVICES=0,1 \
TRAIN_BATCH_SIZE=4 \
OUTPUT_DIR=./runs/single_task_tone_2gpu \
bash scripts/train_single_task_primitive_learning.sh
```

Multi-task run with two groups:

```bash
PRIMITIVE_GROUPS="detail exposure" \
NUM_PROCESSES=1 \
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_DIR=./runs/multi_task_detail_exposure \
bash scripts/train_multi_task_primitive_learning.sh
```

Multi-task run with all four groups:

```bash
PRIMITIVE_GROUPS="detail tone exposure depth" \
NUM_PROCESSES=2 \
CUDA_VISIBLE_DEVICES=0,1 \
TRAIN_BATCH_SIZE=2 \
OUTPUT_DIR=./runs/multi_task_four_groups_2gpu \
bash scripts/train_multi_task_primitive_learning.sh
```

### Direct CLI

Use the package CLI when you need full control:

```bash
PYTHONPATH=src python -m comphoser.cli.train \
  --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-4B \
  --output_dir ./runs/manual_multitask \
  --comphoser_mode lora_qformer \
  --comphoser_primitive_groups detail tone exposure depth \
  --comphoser_data_backend preprocessed \
  --comphoser_validation_mode batch
```

### Important Knobs

The shell launchers expose these environment variables:

- `PRETRAINED_MODEL_NAME_OR_PATH`
- `OUTPUT_DIR`
- `NUM_PROCESSES`
- `CUDA_VISIBLE_DEVICES`
- `COMPHOSER_MODE`
- `COMPHOSER_DATA_BACKEND`
- `COMPHOSER_VALIDATION_MODE`
- `COMPHOSER_GATE_LOSS_WEIGHT`
- `TRAIN_BATCH_SIZE`
- `LEARNING_RATE`
- `MAX_TRAIN_STEPS`
- `VALIDATION_STEPS`
- `NUM_VALIDATION_IMAGES`
- `CHECKPOINTING_STEPS`
- `MIXED_PRECISION`
- `REPORT_TO`

Defaults:

- single-task launcher: `PRIMITIVE_GROUP=detail`, `NUM_PROCESSES=1`
- multi-task launcher: `PRIMITIVE_GROUPS="detail tone exposure depth"`, `NUM_PROCESSES=2`
- both launchers: `COMPHOSER_MODE=lora_qformer`, `COMPHOSER_DATA_BACKEND=preprocessed`, `COMPHOSER_VALIDATION_MODE=batch`

## Check Outputs

Main export location:

```text
{output_dir}/comphoser/
```

Common artifacts:

- `metadata.json`
- `shared_qwp_or_qformer.safetensors`
- `global_query_bank.safetensors`
- `controlled_validation/`

Validation behavior:

- `batch`: runs validation-set inference and writes summary artifacts
- `single`: runs one explicit validation case from `--validation_prompt` and `--validation_image`
- `off`: disables ComPhoser-owned validation

Typical validation outputs:

- `{image_id}_input.png`
- `{image_id}_output_1.png`
- `{image_id}_gt.png`
- `{image_id}_all.png`
- `summary.json`

The validation `summary.json` includes per-output, per-sample, and task-level metric aggregates:

- `psnr_db`: RGB PSNR in dB, higher is better
- `ssim`: mean RGB SSIM, higher is better
- `delta_e_2000`: mean CIEDE2000 Delta E, lower is better
- `qformer_gate_accuracy_pct` and `qformer_gate_loss` in `lora_qformer` mode

Mixed runs also write task-local folders such as:

```text
{output_dir}/comphoser/periodic_validation/{step}/{dataset_id}/
{output_dir}/comphoser/controlled_validation/{dataset_id}/
```

## Evaluate A Checkpoint

Use the package evaluator to run an existing ComPhoser primitive checkpoint on a paired validation split and write a report under `runs/reports/...`.

Example depth/bokeh checkpoint evaluation:

```bash
PYTHONPATH=src python -m comphoser.cli.evaluate_checkpoint \
  --checkpoint_dir {ckpt_path} \
  --dataset_root {data_tgt} \
  --output_dir {output_path} \
  --num_outputs_per_sample 1 \
  --num_inference_steps 8 \
  --seed 17
```

The same entrypoint is available after install as:

```bash
comphoser-evaluate-checkpoint \
  --checkpoint_dir runs/.../checkpoint-100000 \
  --dataset_root data/depth_bokeh__RealBokeh/val \
  --output_dir runs/reports/primitive/depth_bokeh_checkpoint_100000
```

Report layout:

```text
runs/reports/primitive/{run_name}/
  summary.md
  metrics.json
  controlled_validation/
    summary.json
    images/
```

Dataset root input can be either the dataset root, such as `data/depth_bokeh__RealBokeh`, or a split root, such as `data/depth_bokeh__RealBokeh/val`. By default the evaluator uses all samples; pass `--sample_limit` for a smaller smoke run.

## Repository Map

- `src/comphoser/`: ComPhoser package code
- `src/comphoser/cli/train.py`: installable training entrypoint
- `src/comphoser/cli/evaluate_checkpoint.py`: checkpoint evaluation entrypoint
- `src/comphoser/evaluation.py`: primitive checkpoint evaluation orchestration
- `src/comphoser/metrics.py`: PSNR, SSIM, and CIEDE2000 metric helpers
- `scripts/train_single_task_primitive_learning.sh`: single-group launcher
- `scripts/train_multi_task_primitive_learning.sh`: multi-group launcher
- `scripts/build_comphoser_raw_dataset.py`: raw dataset builder
- `scripts/build_comphoser_preprocessed_dataset.py`: latent-cache builder
- `examples/dreambooth/train_dreambooth_lora_flux2_klein_img2img.py`: retained compatibility shim
- `docs/`: architecture, dataset contract, and project state

## Current Status

Implemented:

- FLUX.2 Klein LoRA img2img baseline
- registry-driven dataset routing for `detail`, `tone`, `exposure`, and `depth`
- mixed training with primitive-group-balanced sampling
- fixed-bank `lora_qformer` training path
- controlled validation export with per-task image, metric, and Q-Former gate summaries
- primitive checkpoint evaluation reports with `summary.md`, `metrics.json`, and controlled-validation artifacts

Not implemented yet:

- arbitrary controller families beyond the fixed four-family catalog
- multi-family-per-sample composition training
- broad convergence and quality evaluation campaigns across all four real datasets
- downstream application-task training on top of primitive composition

# ComPhoser

FLUX.2 Klein image-to-image LoRA + a fixed-bank Q-Former controller (`src/comphoser/`) that gates 4 query slots per primitive family (detail / tone / exposure / depth) at the conditioning boundary. Backbone, VAE, tokenizer, and text encoders stay frozen; only the LoRA adapter and the Q-Former are trained.

For project-level status, the open research question, and what each subsystem currently does, read `docs/STATUS.md` first (kept local-only — see [Where things live](#where-things-live)).

---

## Setup

```bash
pip install -e .
pip install -r examples/dreambooth/requirements_flux.txt
accelerate config default
hf auth login            # FLUX.2 Klein is gated
```

Most direct CLI invocations need `PYTHONPATH=src` (the shell launchers set this for you).

---

## Train

Three training modes (`--comphoser_mode`):

- `baseline` — vanilla FLUX.2 Klein img2img LoRA (retained from the diffusers example).
- `lora_only` — ComPhoser dataset/runtime path, no Q-Former.
- `lora_qformer` — adds the fixed-bank controller (4 families {detail, tone, exposure, depth}; the launchers default to 8 query slots/family = 32 tokens via the small controller below).

> **Default controller + recipe (EXP-006/007):** the launchers default to a small, high-quality controller — `routing_dim=1024` bottleneck + `queries_per_primitive=8` + `gate_head_hidden=1024` + `num_layers=1` (~34M / 0.13GB), trained at **lr 1e-4, gate weight 1e-2**. It routes all 4 families and beats the legacy 8.8GB controller on image quality. Override the `COMPHOSER_QFORMER_*` env vars (num_layers=3, routing_dim=0, queries_per_primitive=4, gate_head_hidden=0) to rebuild the legacy controller for old checkpoints.

Two main training strategies for `lora_qformer` (**A** all-in-one, **B** step-by-step), plus an optional identity warmup and a unified downstream finetune. Each named launcher under `scripts/` runs **one** training invocation — chain them yourself by passing the previous run's output_dir via `INIT_FROM_CHECKPOINT`. All launchers source [`scripts/_train_common.sh`](scripts/_train_common.sh) for shared env-var defaults.

| Launcher                                       | Role                                                                                                                       |
|------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| `scripts/train_identity.sh`                    | **Optional identity-preservation warmup — strategy-independent.** `IdentityWrapper` rewrites input = target; image loss alone trains LoRA + Q-Former (no BCE gate loss). Run before **either** strategy and warm-start the next launcher via `INIT_FROM_CHECKPOINT`. Default validation off. |
| `scripts/train_A.sh`                            | **Strategy A — primitives only.** One pass over auto-discovered primitive (catalog) folders only — matches `(detail|tone|exposure|depth)_<task>__<dataset>` under `data/` and pins them via `--train_dataset_ids`. Image loss only; no family concept; no BCE gate loss. |
| `scripts/train_A_full.sh`                       | **Strategy A (full) — primitives + downstream (ablation).** A `train_A.sh` spin-off: one pass over **every** contract folder under `data/` (primitives **and** `downstream_*`) so primitives + downstream are learned jointly instead of pretrain-then-finetune. In all-in-one mode the trainer bypasses the family-catalog gate, so no shell-side enumeration is needed. Image loss only; no BCE gate loss. |
| `scripts/train_lora.sh`                         | **FLUX + LoRA only (no Q-Former) — primitives only.** The `lora_only` counterpart of `train_A.sh`: forces `--comphoser_mode lora_only`, so no controller is built. Trains on the auto-discovered primitive (catalog) folders, **validates over the entire dataset** (every folder with a `val/` split, including `downstream_*`). Image loss only; no gate to supervise (Q-Former env knobs are inert). |
| `scripts/train_lora_full.sh`                    | **FLUX + LoRA only (no Q-Former) — primitives + downstream.** The `lora_only` counterpart of `train_A_full.sh`: forces `--comphoser_mode lora_only`. One all-in-one pass over **every** contract folder under `data/`, and **validates over the entire dataset**. Image loss only; no gate to supervise (Q-Former env knobs are inert). |
| `scripts/train_B.sh`                            | **Strategy B — multi-task primitives.** Group-balanced × dataset-uniform sampling across the 4 primitive families with BCE gate loss. Optionally warm-start from `train_identity.sh` via `INIT_FROM_CHECKPOINT`. |
| `scripts/train_downstream.sh`                  | **Downstream finetune (unified).** One entrypoint for both pretrained inits (A or B) and both downstream versions. Trains an additive `downstream` LoRA from the frozen pretrained controller (`INIT_FROM_CHECKPOINT`). `DOWNSTREAM_MODE=integrated` → one joint LoRA over `DOWNSTREAM_TASKS`; `DOWNSTREAM_MODE=respective` → one LoRA per task (in-process loop) under `OUTPUT_DIR/<dataset_id>/`. No BCE gate loss. |

Unified downstream finetune (the single code path for finetuning either pretrained controller to the downstream tasks):

```bash
# Integrated — one additive LoRA over all four downstream tasks, warm-started from Strategy B.
INIT_FROM_CHECKPOINT=./runs/B_stage2_noS1 DOWNSTREAM_MODE=integrated \
OUTPUT_DIR=./runs/B_downstream_integrated \
  bash scripts/train_downstream.sh

# Respective — a separate additive LoRA per task (in-process loop), warm-started from Strategy A.
INIT_FROM_CHECKPOINT=./runs/A_primitives DOWNSTREAM_MODE=respective \
OUTPUT_DIR=./runs/A_downstream_respective \
DOWNSTREAM_TASKS="downstream_isp__DPED downstream_isp__ZRR downstream_isp__fivek downstream_llie__LOL-v2-Real" \
  bash scripts/train_downstream.sh
# writes ./runs/A_downstream_respective/<dataset_id>/ per task
```

Strategy B with the optional identity warmup, then a downstream finetune (thread `INIT_FROM_CHECKPOINT`):

```bash
# Optional identity warmup (skip to start Strategy B from random init)
OUTPUT_DIR=./runs/expB/identity \
  bash scripts/train_identity.sh

# Strategy B primitive learning, warm-started from the identity warmup
INIT_FROM_CHECKPOINT=./runs/expB/identity OUTPUT_DIR=./runs/expB/primitives \
  bash scripts/train_B.sh

# Downstream finetune (additive LoRA) on a single task, warm-started from Strategy B
INIT_FROM_CHECKPOINT=./runs/expB/primitives OUTPUT_DIR=./runs/expB/downstream_zrr \
DOWNSTREAM_TASKS=downstream_isp__ZRR \
  bash scripts/train_downstream.sh
```

Common knobs (env vars; defaults in the script):

| Var                                 | Purpose                                                                   |
|-------------------------------------|---------------------------------------------------------------------------|
| `PRIMITIVE_GROUPS`                  | Subset of `detail tone exposure depth` (default: all 4)                   |
| `EXCLUDE_DATASET_IDS`               | Space-separated dataset_ids to drop after discovery (e.g. `"detail_blur__RSBlur"`). Ignored when `TRAIN_DATASET_IDS` is set. |
| `TRAIN_DATASET_IDS`                 | Space-separated explicit training pool. When set, **overrides** the family filter + `EXCLUDE_DATASET_IDS`; uses exactly these folders. Non-catalog folders (e.g. `downstream_*`) are loaded via the same direct-load fallback as `DOWNSTREAM_TASKS`. Default unset = use family + exclude path. |
| `VALIDATION_DATASET_IDS`            | Space-separated explicit validation fan-out pool. When set, **overrides** the default (controls.tasks + downstream-target append). Same direct-load fallback for non-catalog folders. Default unset = walk every discovered task plus the downstream target. |
| `DOWNSTREAM_TASKS`                  | `train_downstream.sh` only — space-separated downstream folder names (default: the four `downstream_*` tasks). The training pool + validation fan-out for the unified downstream finetune. A single name yields a one-task finetune (replaces the old per-task Stage 3 launcher). |
| `DOWNSTREAM_MODE`                   | `train_downstream.sh` only — `integrated` (default; one joint additive LoRA over all `DOWNSTREAM_TASKS`) or `respective` (one additive LoRA per task, trained in an in-process loop, written under `OUTPUT_DIR/<dataset_id>/`). |
| `CUDA_VISIBLE_DEVICES` / `NUM_PROCESSES` | GPU selection / `accelerate launch` parallelism                       |
| `TRAIN_BATCH_SIZE`                  | Per-device batch size                                                     |
| `LEARNING_RATE`, `LR_SCHEDULER`     | Optimizer schedule                                                        |
| `MAX_TRAIN_STEPS` / `VALIDATION_STEPS` / `CHECKPOINTING_STEPS` | Per-launcher step budgets (defaults vary per script)            |
| `OUTPUT_DIR` / `RUN_TAG`            | Output dir for this launcher invocation                                    |
| `INIT_FROM_CHECKPOINT`              | Cross-stage warm-start source (typically the previous launcher's OUTPUT_DIR) |
| `COMPHOSER_VALIDATION_MODE`         | `batch` / `single` / `off`. `train_identity.sh` defaults to `off`; the others default to `batch`. |
| `VALIDATION_CHUNK_SIZE`             | Optional: cap each periodic validation to K tasks; the chunk rotates through the discovered fan-out across successive calls so one full cycle covers everything every `ceil(N/K)` validation steps. End-of-training validation always walks every task. |
| `PIVOTAL_VALIDATION_DATASET_ID`     | Optional dataset_id always validated on every periodic call. Auto-added to the fan-out if absent. With `VALIDATION_CHUNK_SIZE=K`, one slot of each chunk is reserved for the pivotal and the remaining tasks cycle through `K-1` slots; with `K=1` only the pivotal runs. Default unset = no pinned task. |
| `VALIDATION_MODEL_CPU_OFFLOAD`      | `auto` (default) / `on` / `off` — policy for diffusers' `enable_model_cpu_offload()` during periodic + end-of-training validation. `auto` skips offload when the validation device's total VRAM is ≥ 48 GiB (FLUX.2 Klein 4B at bf16 peaks ~27 GiB; GPU-resident is ~7× faster) and falls back to offload on smaller GPUs to avoid OOM. `on` always uses offload; `off` always keeps the pipeline GPU-resident. |
| `NUM_VALIDATION_INFERENCE_STEPS`    | Denoising steps for controlled validation (periodic + end-of-training). Launcher default `4` (the bare `--num_validation_inference_steps` default is `8`). Klein is guidance-distilled, so `4` runs ~2× faster and only affects validation image fidelity, not the trained weights. |
| `COMPHOSER_GATE_LOSS_WEIGHT_{INITIAL,FINAL}` | BCE auxiliary weight schedule                                    |
| `COMPHOSER_QFORMER_IMAGE_ROUTING`   | `0` (default) / `1` — `lora_qformer` only. `1` feeds the condition-image latent into Q-Former gate prediction via a learnable attn-pool (output path / BCE unchanged); `0` keeps the prompt-only v1 controller. Toggle for the prompt-only vs image-routing A/B. |
| `COMPHOSER_QFORMER_COND_SUMMARY_TOKENS` | Pooled condition-image summary tokens when image routing is on (default `4`; no-op when off). |

Resume options:

- `--resume_from_checkpoint <dir>`: in-stage continuation — restores model, optimizer, scheduler, and `global_step`.
- `--init_from_checkpoint <dir>`: cross-stage warm-start — loads LoRA + Q-Former weights only; optimizer / LR scheduler / `global_step` reset. The shell wrapper uses this between stages.

**Pretrain-then-finetune workflow (e.g. Strategy A primitives pretrain → downstream finetune):**

```bash
# Pretrain on primitives only (downstream auto-skipped)
OUTPUT_DIR=./runs/pretrain_aio \
  bash scripts/train_A.sh

# Finetune on a single downstream task (one-element DOWNSTREAM_TASKS).
INIT_FROM_CHECKPOINT=./runs/pretrain_aio \
DOWNSTREAM_TASKS=downstream_isp__ZRR \
OUTPUT_DIR=./runs/finetune_zrr \
  bash scripts/train_downstream.sh
```

Direct CLI (skip the wrapper):

```bash
PYTHONPATH=src python -m comphoser.cli.train \
  --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-4B \
  --output_dir ./runs/<name> \
  --comphoser_mode lora_qformer \
  --comphoser_primitive_groups detail tone exposure depth \
  --comphoser_data_backend preprocessed \
  --comphoser_validation_mode batch
```

Controlled-validation summaries (periodic + end-of-training) are written under `<output_dir>/comphoser/` with artifact schema `comphoser-controlled-validation-v9` (PSNR, SSIM, mean CIEDE2000 ΔE, LPIPS-Alex, plus Q-Former gate diagnostics in `lora_qformer`).

---

## Inference & Evaluate

`scripts/infer_primitive_checkpoint.sh` wraps `python -m comphoser.cli.evaluate_checkpoint` (also installed as `comphoser-evaluate-checkpoint`). It runs deterministic inference over a dataset split and writes per-sample artifacts + a summary with PSNR, SSIM, ΔE 2000, and LPIPS-Alex.

```bash
CHECKPOINT_DIR=./runs/<name> \
DATASET_ROOT=data/exposure_ec__MSEC/val \
bash scripts/infer_primitive_checkpoint.sh
```

Outputs land in `runs/reports/primitive/<RUN_TAG>/`. Override `OUTPUT_DIR` to choose a different location.

Key knobs:

| Var                       | Default          | Purpose                                                                                |
|---------------------------|------------------|----------------------------------------------------------------------------------------|
| `EVALUATION_MODE`         | `lora_qformer`   | `lora_qformer` (needs `CHECKPOINT_DIR`) or `flux_only` (no checkpoint, baseline FLUX) |
| `GATING_TYPE`             | `predicted_only` | Q-Former gate override: `predicted_only` / `all_zero` / `all_one` / `detail` / `tone` / `exposure` / `depth` / `custom` |
| `EXPLICIT_TOKEN_MASKING`  |                  | 16 floats; required when `GATING_TYPE=custom`                                          |
| `SAMPLE_LIMIT`            |                  | Cap number of validation samples for quick checks                                      |
| `NUM_INFERENCE_STEPS`     | `8`              | Diffusion sampling steps                                                               |
| `NUM_OUTPUTS_PER_SAMPLE`  | `1`              | Seeds per input image                                                                  |

The same evaluator is invoked automatically at the end of training when `--comphoser_validation_mode batch` and walks every dataset discovered under the active primitive groups (so Strategy B and downstream reports cover all 4 families as a regression check).

---

## Datasets

On-disk layout (one folder per `<group>_<task_variant>__<dataset_name>`, with `train/` and `val/` subdirs):

```
data/<name>/<split>/
  raw/
    images/{input,target}/<id>.jpg
    prompt/<id>.txt
  preprocessed/
    image_latent_cache/{input,target}/<id>.pt
    prompt_latent_cache/<id>.pt
```

Discovery is automatic — any folder following this convention is picked up by `--comphoser_primitive_groups`. Builder scripts live in `scripts/build_*_raw_dataset.py` (raw pairs) and `scripts/build_comphoser_preprocessed_dataset.py` (latent caches). Never overwrite `raw/` in place; rebuild `preprocessed/` after raw changes.

Full spec: `docs/project/dataset_contract.md` (kept local-only).

---

## Tests & lint

```bash
PYTHONPATH=src python -m unittest discover -s tests/comphoser -p 'test_*.py'
ruff check src tests
ruff format src tests
```

---

## Where things live

Committed:

- [`src/comphoser/`](src/comphoser/) — package source (`trainer.py`, `train_args.py`, `qformer.py`, `controls.py`, `datasets.py`, `inference.py`, `evaluation.py`).
- [`scripts/`](scripts/) — shell launchers and dataset builders.

Local-only (gitignored by project convention — present in a working checkout, not in the published repo):

- `tests/comphoser/` — unit + integration tests.
- `docs/STATUS.md` — current status, open research question, board.
- `docs/architecture/training_strategy.md` — A vs B design memo + implementation notes.
- `docs/architecture/qformer_conditioning.md` — v1 controller contract.
- `docs/project/dataset_contract.md` — on-disk dataset format.
- `CLAUDE.md` — operating guide for Claude Code in this repo (architecture summary + invariants).

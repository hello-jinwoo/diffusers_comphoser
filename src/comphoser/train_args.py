"""Package-owned training argument parsing for ComPhoser and the retained trainer."""

from __future__ import annotations

import argparse
import os
from typing import Sequence

from .datasets import COMPHOSER_DATA_BACKENDS
from .inference import DEFAULT_CONTROLLED_VALIDATION_STEPS
from .qformer import (
    DEFAULT_QFORMER_COND_SUMMARY_TOKENS,
    DEFAULT_QFORMER_NUM_LAYERS,
    DEFAULT_QFORMER_QUERY_COUNT,
)
from .training import PILOT_GATE_LOSS_WEIGHT_SCHEDULERS, PILOT_TRAINING_MODES


COMPHOSER_VALIDATION_MODES = ("batch", "single", "off")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--bnb_quantization_config_path",
        type=str,
        default=None,
        help="Quantization config in a JSON file that will be used to define the bitsandbytes quant config of the DiT.",
    )
    parser.add_argument(
        "--do_fp8_training",
        action="store_true",
        help="if we are doing FP8 training.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images "
            "(could be your own, possibly private, dataset). It can also be a path pointing to a local copy of a "
            "dataset in your filesystem, or to a folder containing files that datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help=(
            "The column of the dataset containing the target image. By default, the standard Image Dataset maps "
            "out 'file_name' to 'image'."
        ),
    )
    parser.add_argument(
        "--cond_image_column",
        type=str,
        default=None,
        help="Column in the dataset containing the condition image. Must be specified when performing I2I fine-tuning",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )
    parser.add_argument(
        "--comphoser_mode",
        type=str,
        default="baseline",
        choices=PILOT_TRAINING_MODES,
        help=(
            "Optional ComPhoser pilot mode. 'baseline' keeps the retained trainer path, 'lora_only' uses the "
            "prepared pilot dataset without the Q-Former, and 'lora_qformer' adds the controller."
        ),
    )
    parser.add_argument(
        "--comphoser_primitive_groups",
        type=str,
        nargs="+",
        default=None,
        help="Selected ComPhoser primitive groups for pilot modes, for example: --comphoser_primitive_groups detail",
    )
    parser.add_argument(
        "--comphoser_data_backend",
        type=str,
        default="preprocessed",
        choices=COMPHOSER_DATA_BACKENDS,
        help=(
            "Data backend for ComPhoser pilot modes. 'preprocessed' loads cached latents and prompt embeddings "
            "from split-local preprocessed/ artifacts, while 'raw' keeps runtime image/prompt encoding."
        ),
    )
    parser.add_argument(
        "--comphoser_validation_mode",
        type=str,
        default="batch",
        choices=COMPHOSER_VALIDATION_MODES,
        help=(
            "Validation surface for ComPhoser pilot modes. 'batch' runs deterministic validation-set inference, "
            "'single' reuses --validation_prompt/--validation_image for one explicit case, and 'off' disables "
            "ComPhoser-owned validation."
        ),
    )
    parser.add_argument(
        "--comphoser_qformer_num_queries",
        type=int,
        default=DEFAULT_QFORMER_QUERY_COUNT,
        help=(
            "Number of learned Q-Former query tokens to use in 'lora_qformer' mode. "
            f"v1 uses a fixed global bank of {DEFAULT_QFORMER_QUERY_COUNT} queries."
        ),
    )
    parser.add_argument(
        "--comphoser_qformer_num_layers",
        type=int,
        default=DEFAULT_QFORMER_NUM_LAYERS,
        help=(
            "Number of prompt-routing trunk layers to use in 'lora_qformer' mode. "
            f"The legacy fixed-bank controller depth is {DEFAULT_QFORMER_NUM_LAYERS}."
        ),
    )
    parser.add_argument(
        "--comphoser_qformer_routing_rounds",
        type=int,
        default=1,
        help="Query-routing cross-attention rounds (weight-shared iterative refinement). 1 (default) = single pass.",
    )
    parser.add_argument(
        "--comphoser_qformer_routing_mean_pool",
        action="store_true",
        help="Skip query cross-attention for routing; use the mean-pooled prompt context for gate prediction.",
    )
    parser.add_argument(
        "--comphoser_qformer_queries_per_primitive",
        type=int,
        default=4,
        help=(
            "Query slots per primitive family (4 families). Total query bank = 4 x this. Default 4 (=16 total); "
            "the query count is derived from this, overriding --comphoser_qformer_num_queries."
        ),
    )
    parser.add_argument(
        "--comphoser_qformer_routing_dim",
        type=int,
        default=0,
        help=(
            "Optional routing bottleneck: run the Q-Former routing path (trunk + query attention + "
            "gate head) at this width instead of hidden_size; the output query bank stays at "
            "hidden_size. 0 (default) = no bottleneck (full width). Must divide by num_heads."
        ),
    )
    parser.add_argument(
        "--comphoser_qformer_ffn_multiplier",
        type=int,
        default=4,
        help="Q-Former trunk FFN width multiplier (dim_feedforward = multiplier x routing width). Default 4.",
    )
    parser.add_argument(
        "--comphoser_qformer_gate_head_hidden",
        type=int,
        default=0,
        help="Optional hidden width for a deeper 2-layer gate head (LayerNorm->Linear->GELU->Linear). 0 = default 1-layer head.",
    )
    parser.add_argument(
        "--comphoser_qformer_output_content_mix",
        action="store_true",
        help=(
            "Blend the prompt-attended routing context into the appended output tokens (prompt-adaptive "
            "content) instead of gating a static bank. No-op at init (learned mix scale starts at 0)."
        ),
    )
    parser.add_argument(
        "--comphoser_qformer_image_routing",
        action="store_true",
        help=(
            "Enable condition-image-aware Q-Former routing in 'lora_qformer' mode. The image latent "
            "(cond_tokens) is attention-pooled to a few summary tokens and fed into the gate "
            "predictor alongside the prompt. Default off keeps the prompt-only v1 controller."
        ),
    )
    parser.add_argument(
        "--comphoser_qformer_cond_summary_tokens",
        type=int,
        default=DEFAULT_QFORMER_COND_SUMMARY_TOKENS,
        help=(
            "Number of pooled condition-image summary tokens used when "
            "--comphoser_qformer_image_routing is set "
            f"(default {DEFAULT_QFORMER_COND_SUMMARY_TOKENS})."
        ),
    )
    parser.add_argument(
        "--comphoser_gate_loss_weight",
        type=float,
        default=0.1,
        help="Legacy fixed auxiliary gate-loss weight source for the fixed-bank Q-Former path.",
    )
    parser.add_argument(
        "--comphoser_gate_loss_weight_initial",
        type=float,
        default=None,
        help="Initial auxiliary gate-loss weight for the fixed-bank Q-Former path.",
    )
    parser.add_argument(
        "--comphoser_gate_loss_weight_final",
        type=float,
        default=None,
        help="Final auxiliary gate-loss weight for the fixed-bank Q-Former path.",
    )
    parser.add_argument(
        "--comphoser_gate_loss_weight_scheduler",
        type=str,
        default="linear",
        choices=PILOT_GATE_LOSS_WEIGHT_SCHEDULERS,
        help="Scheduler mode for interpolating the auxiliary gate-loss weight across optimizer steps.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default=None,
        choices=(
            "all_in_one",
            "step_by_step_stage1",
            "step_by_step_stage2",
            "step_by_step_stage3",
            "single_dataset",
            "downstream",
        ),
        help=(
            "Top-level training strategy. Unset (default) keeps the legacy per-group dispatch. "
            "'all_in_one' samples uniformly across all discovered folders (subject to --comphoser_primitive_groups filter). "
            "'step_by_step_stage{1,2,3}' runs one stage of the staged curriculum (see docs/architecture/training_strategy.md). "
            "'single_dataset' trains on the single folder named by --downstream_target_dataset_id. "
            "'downstream' finetunes a NEW additive 'downstream' LoRA on top of the frozen pretrained "
            "LoRA + Q-Former (warm-started via --init_from_checkpoint, e.g. runs/A_primitives or "
            "runs/B_stage2_noS1) over the folders named by --downstream_target_dataset_ids; "
            "--downstream_mode selects integrated (one joint LoRA) vs respective (one LoRA per task)."
        ),
    )
    parser.add_argument(
        "--downstream_target_dataset_id",
        type=str,
        default=None,
        help=(
            "Required for --training_strategy=step_by_step_stage3 and --training_strategy=single_dataset: "
            "the dataset_id (folder name) of the single dataset to train on. If the name does not "
            "match any folder discovered for the selected --comphoser_primitive_groups (e.g. a "
            "non-catalog 'downstream_*' folder), the trainer falls back to loading data/<name>/ "
            "directly. Also accepted by --training_strategy=downstream as a single-target shorthand "
            "(equivalent to a one-element --downstream_target_dataset_ids)."
        ),
    )
    parser.add_argument(
        "--downstream_target_dataset_ids",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Required for --training_strategy=downstream: one or more dataset_ids (folder names, "
            "typically non-catalog 'downstream_*' folders) to finetune the additive 'downstream' "
            "LoRA on. With --downstream_mode=integrated a single LoRA is trained jointly over all "
            "of them (uniform-per-folder sampling); with --downstream_mode=respective the trainer "
            "loops in-process and trains a separate LoRA per id into <output_dir>/<dataset_id>/. "
            "Each name is resolved against the discovered catalog first, falling back to data/<name>/."
        ),
    )
    parser.add_argument(
        "--downstream_mode",
        type=str,
        default="integrated",
        choices=("integrated", "respective"),
        help=(
            "Only valid with --training_strategy=downstream. 'integrated' trains ONE additive "
            "downstream LoRA jointly over every --downstream_target_dataset_ids folder. 'respective' "
            "trains them one at a time (in-process loop), producing one additive LoRA per task under "
            "<output_dir>/<dataset_id>/. Default 'integrated'."
        ),
    )
    parser.add_argument(
        "--exclude_dataset_ids",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional list of dataset_ids (folder names) to exclude from training after "
            "auto-discovery and the --comphoser_primitive_groups filter. Useful for ablation "
            "studies or pretraining phases that should skip specific primitives. Excluding does "
            "not affect the validation fan-out (which still walks every discovered task). "
            "Ignored when --train_dataset_ids is set."
        ),
    )
    parser.add_argument(
        "--train_dataset_ids",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional explicit allow-list of dataset_ids for training. When set it acts as the "
            "exhaustive training task pool, bypassing --comphoser_primitive_groups and "
            "--exclude_dataset_ids. Each name is resolved against the discovered catalog first; "
            "if not found, falls back to loading data/<name>/ directly (lets non-catalog "
            "downstream_* folders participate). Default unset = use the discovery + exclude path."
        ),
    )
    parser.add_argument(
        "--validation_dataset_ids",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional explicit allow-list of dataset_ids for the validation fan-out. When set it "
            "acts as the exhaustive validation task pool, bypassing the default "
            "(controls.tasks + --downstream_target_dataset_id). Each name is resolved against "
            "the discovered catalog first; if not found, falls back to data/<name>/. Default "
            "unset = walk every discovered task plus the downstream target."
        ),
    )
    parser.add_argument(
        "--stage1_identity_prompt_mix_ratio",
        type=float,
        default=0.5,
        help=(
            "Stage 1 only: fraction of identity-pretraining samples that use the 'preserve' prompt vs an empty prompt. "
            "Default 0.5."
        ),
    )
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=False,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        help="path to an image that is used during validation as the condition image to verify that the model is learning.",
    )
    parser.add_argument(
        "--skip_final_inference",
        default=False,
        action="store_true",
        help=(
            "Whether to skip the final inference step with loaded lora weights upon training completion. This will "
            "run intermediate validation inference if `validation_prompt` is provided. Specify to reduce memory."
        ),
    )
    parser.add_argument(
        "--final_validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during a final validation to verify that the model is learning. Ignored if `--validation_prompt` is provided.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help=(
            "Baseline validation uses this as the number of generated images. ComPhoser batch validation uses this "
            "as the number of validation input samples to process."
        ),
    )
    parser.add_argument(
        "--num_validation_seeds_per_image",
        type=int,
        default=2,
        help=(
            "For ComPhoser validation, number of generated outputs to save per validation sample using deterministic "
            "seed variation."
        ),
    )
    parser.add_argument(
        "--num_validation_inference_steps",
        type=int,
        default=DEFAULT_CONTROLLED_VALIDATION_STEPS,
        help=(
            "Number of denoising steps for ComPhoser controlled validation (periodic + "
            "end-of-training). FLUX.2 Klein is guidance-distilled, so fewer steps (e.g. 4) run "
            "roughly proportionally faster while only affecting validation image fidelity, not "
            f"the trained weights. Default {DEFAULT_CONTROLLED_VALIDATION_STEPS}."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X steps. Dreambooth validation consists of running the prompt "
            "`args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--validation_chunk_size",
        type=int,
        default=None,
        help=(
            "Optional: for periodic ComPhoser validation, restrict each call to this many tasks "
            "and cycle through the fan-out list across successive validation steps. With N "
            "discovered tasks and chunk size K, one full rotation covers everything every ceil(N/K) "
            "validation calls. Default (None / 0 / >= N) = walk every task each periodic call. "
            "End-of-training validation always walks every task regardless of this flag."
        ),
    )
    parser.add_argument(
        "--pivotal_validation_dataset_id",
        type=str,
        default=None,
        help=(
            "Optional dataset_id (folder name) that is always validated on every periodic call. "
            "Auto-added to the fan-out if not already present (resolved against the discovered "
            "catalog with fallback to data/<id>/). When --validation_chunk_size=K is also set, "
            "one slot of each chunk is reserved for the pivotal task and the remaining tasks "
            "cycle through K-1 slots; with K=1 only the pivotal task runs each periodic call. "
            "Default unset = no pinned task."
        ),
    )
    parser.add_argument(
        "--validation_model_cpu_offload",
        type=str,
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Policy for diffusers' enable_model_cpu_offload() during periodic + "
            "end-of-training validation. auto (default) skips offload when the validation "
            "device's total VRAM is >= 48 GiB (FLUX.2 Klein 4B at bf16 peaks ~27 GiB; "
            "GPU-resident is ~7x faster than offload) and falls back to offload on smaller "
            "GPUs to avoid OOM. on = always offload. off = always GPU-resident."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=4,
        help="LoRA alpha to be used for additional scaling.",
    )
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout probability for LoRA layers")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-dreambooth-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "A seed for reproducible training and validation. Defaults to 42 so runs are deterministic "
            "out of the box (samplers, augmentation, and validation generators all key off it). Pass an "
            "explicit value per arm of an A/B comparison."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this "
            "resolution"
        ),
    )
    parser.add_argument(
        "--aspect_ratio_buckets",
        type=str,
        default=None,
        help=(
            "Aspect ratio buckets to use for training. Define as a string of 'h1,w1;h2,w2;...'. e.g. "
            "'1024,1024;768,1360;1360,768;880,1168;1168,880;1248,832;832,1248' Images will be resized and cropped "
            "to fit the nearest bucket. If provided, --resolution is ignored."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly "
            "cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final "
            "checkpoints in case they are better than the last checkpoint, and are also suitable for resuming "
            "training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by "
            '`--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--init_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Warm-start from a checkpoint directory: load LoRA + Q-Former weights only, with a fresh "
            "optimizer, LR scheduler, and global step counter. Distinct from --resume_from_checkpoint "
            "(which continues mid-training and restores all state). Use this to chain training stages "
            "(e.g. Stage 2 starting from Stage 1's final checkpoint)."
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", '
            '"polynomial", "constant", "constant_with_warmup"]. In ComPhoser training entrypoints, '
            '"constant" applies warmup first when --lr_warmup_steps is positive.'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help='We default to the "none" weighting scheme for uniform sampling and uniform loss',
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `logit_normal` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `logit_normal` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `mode` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help='The optimizer type to use. Choose between ["AdamW", "prodigy"]',
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help=(
            "coefficients for computing the Prodigy stepsize using running averages. If set to None, uses the value "
            "of square root of beta2. Ignored if optimizer is adamW"
        ),
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            "The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. "
            'E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to "
            "*output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, "
            "see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"` '
            '(default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= "
            "1.10.and an Nvidia Ampere GPU. Default to the value of accelerate config of the current system or the "
            "flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--distributed_timeout_seconds",
        type=int,
        default=3600,
        help=(
            "Timeout for distributed collectives. Increase this when rank-0-only validation or checkpoint work can "
            "keep the other ranks waiting for longer than the default process-group timeout."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Whether to offload the VAE and the text encoder to CPU when they are not used.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--enable_npu_flash_attention", action="store_true", help="Enabla Flash Attention for NPU")
    parser.add_argument("--fsdp_text_encoder", action="store_true", help="Use FSDP for text encoder")
    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.comphoser_mode == "baseline":
        if args.cond_image_column is None:
            raise ValueError(
                "you must provide --cond_image_column for image-to-image training. Otherwise please see Flux2 text-to-image training example."
            )
        assert args.image_column is not None
        assert args.caption_column is not None

        if args.dataset_name is None and args.instance_data_dir is None:
            raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

        if args.dataset_name is not None and args.instance_data_dir is not None:
            raise ValueError("Specify only one of `--dataset_name` or `--instance_data_dir`")

        if (
            args.validation_prompt is not None or args.final_validation_prompt is not None
        ) and args.validation_image is None:
            raise ValueError(
                "Baseline validation requires --validation_image whenever a validation prompt is provided"
            )
    else:
        if not args.comphoser_primitive_groups:
            raise ValueError("ComPhoser pilot modes require --comphoser_primitive_groups")
        if args.comphoser_mode == "lora_qformer" and args.comphoser_qformer_queries_per_primitive <= 0:
            raise ValueError("--comphoser_qformer_queries_per_primitive must be positive in lora_qformer mode")
        # The query count is derived from queries_per_primitive (4 families x qpf); the legacy fixed-16
        # requirement only applies at the default qpf=4.
        if (
            args.comphoser_mode == "lora_qformer"
            and args.comphoser_qformer_queries_per_primitive == 4
            and args.comphoser_qformer_num_queries != DEFAULT_QFORMER_QUERY_COUNT
        ):
            raise ValueError(
                f"--comphoser_qformer_num_queries must stay {DEFAULT_QFORMER_QUERY_COUNT} at the default "
                "queries_per_primitive=4"
            )
        if args.comphoser_mode == "lora_qformer" and args.comphoser_qformer_num_layers <= 0:
            raise ValueError("--comphoser_qformer_num_layers must be positive in lora_qformer mode")
        if args.comphoser_qformer_image_routing and args.comphoser_mode != "lora_qformer":
            raise ValueError("--comphoser_qformer_image_routing is only valid in lora_qformer mode")
        if args.comphoser_mode == "lora_qformer" and args.comphoser_qformer_cond_summary_tokens <= 0:
            raise ValueError("--comphoser_qformer_cond_summary_tokens must be positive in lora_qformer mode")
        if args.comphoser_gate_loss_weight < 0.0:
            raise ValueError("--comphoser_gate_loss_weight must be non-negative")
        if args.comphoser_gate_loss_weight_initial is None:
            args.comphoser_gate_loss_weight_initial = float(args.comphoser_gate_loss_weight)
        if args.comphoser_gate_loss_weight_final is None:
            args.comphoser_gate_loss_weight_final = float(args.comphoser_gate_loss_weight)
        if args.comphoser_gate_loss_weight_initial < 0.0:
            raise ValueError("--comphoser_gate_loss_weight_initial must be non-negative")
        if args.comphoser_gate_loss_weight_final < 0.0:
            raise ValueError("--comphoser_gate_loss_weight_final must be non-negative")
        if args.comphoser_gate_loss_weight_scheduler == "logarithmic" and (
            args.comphoser_gate_loss_weight_initial <= 0.0 or args.comphoser_gate_loss_weight_final <= 0.0
        ):
            raise ValueError(
                "--comphoser_gate_loss_weight_scheduler=logarithmic requires positive "
                "--comphoser_gate_loss_weight_initial and --comphoser_gate_loss_weight_final"
            )
        args.comphoser_gate_loss_weight = float(args.comphoser_gate_loss_weight_initial)
        if args.num_validation_seeds_per_image <= 0:
            raise ValueError("--num_validation_seeds_per_image must be positive")
        if args.num_validation_inference_steps <= 0:
            raise ValueError("--num_validation_inference_steps must be positive")
        if args.comphoser_validation_mode == "batch" and args.num_validation_images <= 0:
            raise ValueError("ComPhoser batch validation requires --num_validation_images to be positive")
        if args.comphoser_validation_mode == "single":
            if args.validation_image is None:
                raise ValueError("ComPhoser single validation requires --validation_image")
            if args.validation_prompt is None and args.final_validation_prompt is None:
                raise ValueError(
                    "ComPhoser single validation requires --validation_prompt or --final_validation_prompt"
                )

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.distributed_timeout_seconds <= 0:
        raise ValueError("--distributed_timeout_seconds must be positive")

    if args.init_from_checkpoint and args.resume_from_checkpoint:
        raise ValueError(
            "--init_from_checkpoint and --resume_from_checkpoint are mutually exclusive. "
            "Use --init_from_checkpoint for cross-stage warm-start (weights only) and "
            "--resume_from_checkpoint for in-stage continuation (full state restore)."
        )

    # --training_strategy related validation
    if not 0.0 <= args.stage1_identity_prompt_mix_ratio <= 1.0:
        raise ValueError("--stage1_identity_prompt_mix_ratio must be within [0.0, 1.0]")

    strategies_requiring_target = ("step_by_step_stage3", "single_dataset")
    if args.training_strategy in strategies_requiring_target and not args.downstream_target_dataset_id:
        raise ValueError(
            f"--training_strategy={args.training_strategy} requires --downstream_target_dataset_id "
            "(the dataset_id of the target folder)"
        )

    # --downstream_target_dataset_ids / --downstream_mode are exclusive to the 'downstream' strategy.
    if args.downstream_target_dataset_ids and args.training_strategy != "downstream":
        raise ValueError("--downstream_target_dataset_ids is only valid with --training_strategy=downstream")

    if args.training_strategy == "downstream":
        # Accept either the plural list or the singular shorthand; normalize into the list so
        # the trainer always reads one canonical, deduped tuple. integrated trains one joint LoRA
        # over all ids; respective loops over the ids one at a time.
        combined = list(args.downstream_target_dataset_ids or ())
        if args.downstream_target_dataset_id:
            combined.append(args.downstream_target_dataset_id)
        combined = list(dict.fromkeys(name for name in combined if name))
        if not combined:
            raise ValueError(
                "--training_strategy=downstream requires --downstream_target_dataset_ids "
                "(one or more folder names) or --downstream_target_dataset_id"
            )
        args.downstream_target_dataset_ids = tuple(combined)
        # The singular field is only meaningful for a one-task run; clear it for multi-task.
        args.downstream_target_dataset_id = combined[0] if len(combined) == 1 else None
    else:
        # Non-downstream strategies never carry a downstream id list.
        args.downstream_target_dataset_ids = None

    if args.downstream_target_dataset_id and args.training_strategy not in (
        *strategies_requiring_target,
        "downstream",
    ):
        raise ValueError(
            "--downstream_target_dataset_id is only valid with --training_strategy="
            "step_by_step_stage3, single_dataset, or downstream"
        )

    return args


def parse_args(input_args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(list(input_args) if input_args is not None else None)
    return validate_args(args)


__all__ = ["COMPHOSER_VALIDATION_MODES", "build_parser", "parse_args", "validate_args"]

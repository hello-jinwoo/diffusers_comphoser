"""Package-owned training argument parsing for ComPhoser and the retained trainer."""

from __future__ import annotations

import argparse
import os
from typing import Sequence

from .datasets import COMPHOSER_DATA_BACKENDS
from .qformer import DEFAULT_QFORMER_QUERY_COUNT
from .training import PILOT_TRAINING_MODES

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
        "--comphoser_run_mode",
        dest="comphoser_mode",
        type=str,
        choices=PILOT_TRAINING_MODES,
        help=argparse.SUPPRESS,
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
        help="Number of learned Q-Former query tokens to use in 'lora_qformer' mode.",
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
        "--validation_steps",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X steps. Dreambooth validation consists of running the prompt "
            "`args.validation_prompt` multiple times: `args.num_validation_images`."
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
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
            "`--checkpointing_steps`, or `\"latest\"` to automatically select the last available checkpoint."
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
            '"polynomial", "constant", "constant_with_warmup"]'
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

        if (args.validation_prompt is not None or args.final_validation_prompt is not None) and args.validation_image is None:
            raise ValueError("Baseline validation requires --validation_image whenever a validation prompt is provided")
    else:
        if not args.comphoser_primitive_groups:
            raise ValueError("ComPhoser pilot modes require --comphoser_primitive_groups")
        if args.comphoser_mode == "lora_qformer" and args.comphoser_qformer_num_queries <= 0:
            raise ValueError("--comphoser_qformer_num_queries must be positive in lora_qformer mode")
        if args.num_validation_seeds_per_image <= 0:
            raise ValueError("--num_validation_seeds_per_image must be positive")
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

    return args


def parse_args(input_args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(list(input_args) if input_args is not None else None)
    return validate_args(args)


__all__ = ["COMPHOSER_VALIDATION_MODES", "build_parser", "parse_args", "validate_args"]

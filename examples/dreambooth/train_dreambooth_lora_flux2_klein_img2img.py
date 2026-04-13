#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "diffusers @ git+https://github.com/huggingface/diffusers.git",
#     "torch>=2.0.0",
#     "accelerate>=0.31.0",
#     "transformers>=4.41.2",
#     "ftfy",
#     "tensorboard",
#     "Jinja2",
#     "peft>=0.11.1",
#     "sentencepiece",
#     "torchvision",
#     "datasets",
#     "bitsandbytes",
#     "prodigyopt",
# ]
# ///

import copy
import itertools
import json
import logging
import math
import os
import random
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

# Prefer this checkout's src/ tree over other editable comphoser installs.
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if _REPO_SRC.is_dir():
    _repo_src_str = str(_REPO_SRC)
    if _repo_src_str in sys.path:
        sys.path.remove(_repo_src_str)
    sys.path.insert(0, _repo_src_str)

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, prepare_model_for_kbit_training, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

import diffusers
from comphoser import (
    PreparedPilotDataset,
    collate_prepared_pilot_examples,
    has_pilot_qformer_checkpoint,
    load_pilot_qformer_checkpoint,
    prepare_pilot_transformer_conditioning,
    resolve_pilot_batch_task_strengths,
    save_pilot_qformer_checkpoint,
)
from comphoser.train_args import parse_args
from comphoser.train_runtime import (
    build_detached_validation_pipeline,
    build_detached_validation_qformer,
    build_pilot_checkpoint_metadata,
    build_pilot_qformer,
    resolve_and_log_pilot_training,
    run_comphoser_validation,
    run_final_comphoser_export,
)
from diffusers import (
    AutoencoderKLFlux2,
    BitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers.training_utils import (
    _collate_lora_metadata,
    _to_cpu_contiguous,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    find_nearest_bucket,
    free_memory,
    get_fsdp_kwargs_from_accelerator,
    offload_models,
    parse_buckets_string,
    wrap_with_fsdp,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
    load_image,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module


if getattr(torch, "distributed", None) is not None:
    import torch.distributed as dist

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.38.0.dev0")

logger = get_logger(__name__)


def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = None,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
    fp8_training=False,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = f"""
# Flux.2 [Klein] DreamBooth LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} DreamBooth LoRA weights for {base_model}.

The weights were trained using [DreamBooth](https://dreambooth.github.io/) with the [Flux2 diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux2.md).

FP8 training? {fp8_training}

## Trigger words

You should use `{instance_prompt}` to trigger the image generation.

## Download model

[Download the *.safetensors LoRA]({repo_id}/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import AutoPipelineForText2Image
import torch
pipeline = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.2", torch_dtype=torch.bfloat16).to('cuda')
pipeline.load_lora_weights('{repo_id}', weight_name='pytorch_lora_weights.safetensors')
image = pipeline('{validation_prompt if validation_prompt else instance_prompt}').images[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/black-forest-labs/FLUX.2/blob/main/LICENSE.md).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "lora",
        "flux2",
        "flux2-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    torch_dtype,
    is_final_validation=False,
):
    args.num_validation_images = args.num_validation_images if args.num_validation_images else 1
    prompt_label = pipeline_args.get("prompt_label", args.validation_prompt)
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {prompt_label}."
    )
    pipeline = pipeline.to(dtype=torch_dtype)
    pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()

    images = []
    for _ in range(args.num_validation_images):
        with autocast_ctx:
            image = pipeline(
                image=pipeline_args["image"],
                prompt_embeds=pipeline_args["prompt_embeds"],
                negative_prompt_embeds=pipeline_args["negative_prompt_embeds"],
                generator=generator,
            ).images[0]
            images.append(image)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {prompt_label}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    free_memory()

    return images

def module_filter_fn(mod: torch.nn.Module, fqn: str):
    # don't convert the output module
    if fqn == "proj_out":
        return False
    # don't convert linear modules with weight dimensions not divisible by 16
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    return True


def resolve_dataset_image_entry(entry: Any, dataset_name: str, split: str = "train") -> Image.Image:
    if isinstance(entry, Image.Image):
        return entry

    image_path = None
    if isinstance(entry, str):
        image_path = entry
    elif isinstance(entry, dict) and isinstance(entry.get("path"), str):
        image_path = entry["path"]

    if image_path is None:
        raise TypeError(
            "Expected the dataset image entry to be a PIL image, a string path, or a {'path': ...} mapping, "
            f"but received {type(entry).__name__}."
        )

    if image_path.startswith(("http://", "https://")):
        return load_image(image_path)

    candidate_paths = [Path(image_path).expanduser()]
    dataset_root = Path(dataset_name).expanduser()
    candidate_paths.append(dataset_root / image_path)
    candidate_paths.append(dataset_root / split / image_path)

    for candidate_path in candidate_paths:
        if candidate_path.is_file():
            return Image.open(candidate_path)

    raise FileNotFoundError(
        f"Could not resolve dataset image path '{image_path}' from dataset root '{dataset_root}'."
    )

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        size=1024,
        repeats=1,
        center_crop=False,
        buckets=None,
    ):
        self.size = size
        self.center_crop = center_crop

        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None

        self.buckets = buckets

        # if --dataset_name is provided or a metadata jsonl file is provided in the local --instance_data directory,
        # we load the training data using load_dataset
        if args.dataset_name is not None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "You are trying to load your data using the datasets library. If you wish to train using custom "
                    "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                    "local folder containing images only, specify --instance_data_dir instead."
                )
            # Downloading and loading a dataset from the hub.
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
            )
            # Preprocessing the datasets.
            column_names = dataset["train"].column_names

            # 6. Get the column names for input/target.
            if args.cond_image_column is not None and args.cond_image_column not in column_names:
                raise ValueError(
                    f"`--cond_image_column` value '{args.cond_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )
            if args.image_column is None:
                image_column = column_names[0]
                logger.info(f"image column defaulting to {image_column}")
            else:
                image_column = args.image_column
                if image_column not in column_names:
                    raise ValueError(
                        f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
            instance_images = dataset["train"][image_column]
            cond_images = None
            cond_image_column = args.cond_image_column
            if cond_image_column is not None:
                cond_images = [
                    resolve_dataset_image_entry(dataset["train"][i][cond_image_column], args.dataset_name)
                    for i in range(len(dataset["train"]))
                ]
                assert len(instance_images) == len(cond_images)

            if args.caption_column is None:
                logger.info(
                    "No caption column provided, defaulting to instance_prompt for all images. If your dataset "
                    "contains captions/prompts for the images, make sure to specify the "
                    "column as --caption_column"
                )
                self.custom_instance_prompts = None
            else:
                if args.caption_column not in column_names:
                    raise ValueError(
                        f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
                custom_instance_prompts = dataset["train"][args.caption_column]
                # create final list of captions according to --repeats
                self.custom_instance_prompts = []
                for caption in custom_instance_prompts:
                    self.custom_instance_prompts.extend(itertools.repeat(caption, repeats))
        else:
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError("Instance images root doesn't exists.")

            instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir())]
            self.custom_instance_prompts = None

        self.instance_images = []
        self.cond_images = []
        for i, img in enumerate(instance_images):
            self.instance_images.extend(itertools.repeat(img, repeats))
            if args.dataset_name is not None and cond_images is not None:
                self.cond_images.extend(itertools.repeat(cond_images[i], repeats))

        self.pixel_values = []
        self.cond_pixel_values = []
        for i, image in enumerate(self.instance_images):
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            dest_image = None
            if self.cond_images:  # todo: take care of max area for buckets
                dest_image = self.cond_images[i]
                image_width, image_height = dest_image.size
                if image_width * image_height > 1024 * 1024:
                    dest_image = Flux2ImageProcessor._resize_to_target_area(dest_image, 1024 * 1024)
                    image_width, image_height = dest_image.size

                multiple_of = 2 ** (4 - 1)  # 2 ** (len(vae.config.block_out_channels) - 1), temp!
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                image_processor = Flux2ImageProcessor()
                dest_image = image_processor.preprocess(
                    dest_image, height=image_height, width=image_width, resize_mode="crop"
                )
                # Convert back to PIL
                dest_image = dest_image.squeeze(0)
                if dest_image.min() < 0:
                    dest_image = (dest_image + 1) / 2
                dest_image = (torch.clamp(dest_image, 0, 1) * 255).byte().cpu()

                if dest_image.shape[0] == 1:
                    # Gray scale image
                    dest_image = Image.fromarray(dest_image.squeeze().numpy(), mode="L")
                else:
                    # RGB scale image: (C, H, W) -> (H, W, C)
                    dest_image = TF.to_pil_image(dest_image)

                dest_image = exif_transpose(dest_image)
                if not dest_image.mode == "RGB":
                    dest_image = dest_image.convert("RGB")

            width, height = image.size

            # Find the closest bucket
            bucket_idx = find_nearest_bucket(height, width, self.buckets)
            target_height, target_width = self.buckets[bucket_idx]
            self.size = (target_height, target_width)

            # based on the bucket assignment, define the transformations
            image, dest_image = self.paired_transform(
                image,
                dest_image=dest_image,
                size=self.size,
                center_crop=args.center_crop,
                random_flip=args.random_flip,
            )
            self.pixel_values.append((image, bucket_idx))
            if dest_image is not None:
                self.cond_pixel_values.append((dest_image, bucket_idx))

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image, bucket_idx = self.pixel_values[index % self.num_instance_images]
        example["instance_images"] = instance_image
        example["bucket_idx"] = bucket_idx
        if self.cond_pixel_values:
            dest_image, _ = self.cond_pixel_values[index % self.num_instance_images]
            example["cond_images"] = dest_image

        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            if caption:
                example["instance_prompt"] = caption
            else:
                example["instance_prompt"] = self.instance_prompt

        else:  # custom prompts were provided, but length does not match size of image dataset
            example["instance_prompt"] = self.instance_prompt

        return example

    def paired_transform(self, image, dest_image=None, size=(224, 224), center_crop=False, random_flip=False):
        # 1. Resize (deterministic)
        resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        image = resize(image)
        if dest_image is not None:
            dest_image = resize(dest_image)

        # 2. Crop: either center or SAME random crop
        if center_crop:
            crop = transforms.CenterCrop(size)
            image = crop(image)
            if dest_image is not None:
                dest_image = crop(dest_image)
        else:
            # get_params returns (i, j, h, w)
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=size)
            image = TF.crop(image, i, j, h, w)
            if dest_image is not None:
                dest_image = TF.crop(dest_image, i, j, h, w)

        # 3. Random horizontal flip with the SAME coin flip
        if random_flip:
            do_flip = random.random() < 0.5
            if do_flip:
                image = TF.hflip(image)
                if dest_image is not None:
                    dest_image = TF.hflip(dest_image)

        # 4. ToTensor + Normalize (deterministic)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.5], [0.5])
        image = normalize(to_tensor(image))
        if dest_image is not None:
            dest_image = normalize(to_tensor(dest_image))

        return (image, dest_image) if dest_image is not None else (image, None)


def collate_fn(examples):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts}
    if any("cond_images" in example for example in examples):
        cond_pixel_values = [example["cond_images"] for example in examples]
        cond_pixel_values = torch.stack(cond_pixel_values)
        cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()
        batch.update({"cond_pixel_values": cond_pixel_values})
    return batch


class BucketBatchSampler(BatchSampler):
    def __init__(self, dataset: Dataset, batch_size: int, drop_last: bool = False):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got drop_last={}".format(drop_last))

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Group indices by bucket
        self.bucket_indices = [[] for _ in range(len(self.dataset.buckets))]
        if hasattr(self.dataset, "bucket_indices"):
            dataset_bucket_indices = tuple(int(bucket_idx) for bucket_idx in self.dataset.bucket_indices)
        elif hasattr(self.dataset, "pixel_values"):
            dataset_bucket_indices = tuple(bucket_idx for _, bucket_idx in self.dataset.pixel_values)
        else:
            raise AttributeError("BucketBatchSampler requires the dataset to expose bucket_indices or pixel_values")

        for idx, bucket_idx in enumerate(dataset_bucket_indices):
            self.bucket_indices[bucket_idx].append(idx)

        self.sampler_len = 0
        self.batches = []

        # Pre-generate batches for each bucket
        for indices_in_bucket in self.bucket_indices:
            # Shuffle indices within the bucket
            random.shuffle(indices_in_bucket)
            # Create batches
            for i in range(0, len(indices_in_bucket), self.batch_size):
                batch = indices_in_bucket[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue  # Skip partial batch if drop_last is True
                self.batches.append(batch)
                self.sampler_len += 1  # Count the number of batches

    def __iter__(self):
        # Shuffle the order of the batches each epoch
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return self.sampler_len


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    if args.do_fp8_training:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    comphoser_training = resolve_and_log_pilot_training(args, logger)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load the tokenizers
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        revision=args.revision,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(accelerator.device)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        accelerator.device
    )

    quantization_config = None
    if args.bnb_quantization_config_path is not None:
        with open(args.bnb_quantization_config_path, "r") as f:
            config_kwargs = json.load(f)
            if "load_in_4bit" in config_kwargs and config_kwargs["load_in_4bit"]:
                config_kwargs["bnb_4bit_compute_dtype"] = weight_dtype
        quantization_config = BitsAndBytesConfig(**config_kwargs)

    transformer = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        quantization_config=quantization_config,
        torch_dtype=weight_dtype,
    )
    if args.bnb_quantization_config_path is not None:
        transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)

    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder.requires_grad_(False)

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            transformer.set_attention_backend("_native_npu")
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu device ")

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    to_kwargs = {"dtype": weight_dtype, "device": accelerator.device} if not args.offload else {"dtype": weight_dtype}
    # flux vae is stable in bf16 so load it in weight_dtype to reduce memory
    vae.to(**to_kwargs)
    # we never offload the transformer to CPU, so we can just use the accelerator device
    transformer_to_kwargs = (
        {"device": accelerator.device}
        if args.bnb_quantization_config_path is not None
        else {"device": accelerator.device, "dtype": weight_dtype}
    )

    is_fsdp = getattr(accelerator.state, "fsdp_plugin", None) is not None
    if not is_fsdp:
        transformer.to(**transformer_to_kwargs)

    if args.do_fp8_training:
        convert_to_float8_training(
            transformer, module_filter_fn=module_filter_fn, config=Float8LinearConfig(pad_inner_dim=True)
        )

    text_encoder.to(**to_kwargs)
    # Initialize a text encoding pipeline and keep it to CPU for now.
    text_encoding_pipeline = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
        revision=args.revision,
    )

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        # target_modules = ["to_k", "to_q", "to_v", "to_out.0"] # just train transformer_blocks

        # train transformer_blocks and single_transformer_blocks
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"] + [
            "to_qkv_mlp_proj",
            *[f"single_transformer_blocks.{i}.attn.to_out" for i in range(24)],
        ]

    # now we will add new LoRA weights the transformer layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)
    qformer = build_pilot_qformer(
        transformer,
        comphoser_training=comphoser_training,
        logger=logger,
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    transformer_cls = type(unwrap_model(transformer))
    qformer_cls = type(unwrap_model(qformer)) if qformer is not None else None

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        modules_to_save: dict[str, Any] = {}
        transformer_model = None
        qformer_model = None
        transformer_weight_index = None
        qformer_weight_index = None

        for index, model in enumerate(models):
            if isinstance(unwrap_model(model), transformer_cls):
                transformer_model = model
                modules_to_save["transformer"] = model
                transformer_weight_index = index
            elif qformer_cls is not None and isinstance(unwrap_model(model), qformer_cls):
                qformer_model = model
                qformer_weight_index = index
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        if transformer_model is None:
            raise ValueError("No transformer model found in 'models'")

        state_dict = accelerator.get_state_dict(transformer_model) if is_fsdp else None

        transformer_lora_layers_to_save = None
        if accelerator.is_main_process:
            peft_kwargs = {}
            if is_fsdp:
                peft_kwargs["state_dict"] = state_dict

            transformer_lora_layers_to_save = get_peft_model_state_dict(
                unwrap_model(transformer_model) if is_fsdp else transformer_model,
                **peft_kwargs,
            )

            if is_fsdp:
                transformer_lora_layers_to_save = _to_cpu_contiguous(transformer_lora_layers_to_save)

            # Remove the transformer/controller weights so that Accelerate does not also save full model checkpoints.
            for weight_index in sorted(
                (
                    index
                    for index in (transformer_weight_index, qformer_weight_index)
                    if index is not None and index < len(weights)
                ),
                reverse=True,
            ):
                weights.pop(weight_index)

            Flux2KleinPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )
            if qformer_model is not None:
                if comphoser_checkpoint_metadata is None:
                    raise ValueError("Missing ComPhoser checkpoint metadata for Q-Former save")
                qformer_to_save = unwrap_model(qformer_model)
                qformer_state_dict = (
                    accelerator.get_state_dict(qformer_model) if is_fsdp else qformer_to_save.state_dict()
                )
                save_pilot_qformer_checkpoint(
                    output_dir,
                    qformer=qformer_to_save,
                    metadata=comphoser_checkpoint_metadata,
                    state_dict=qformer_state_dict,
                )

    def load_model_hook(models, input_dir):
        transformer_ = None
        qformer_ = None
        has_stage3c_qformer_checkpoint = qformer_cls is not None and has_pilot_qformer_checkpoint(input_dir)

        if not is_fsdp:
            for model_index in range(len(models) - 1, -1, -1):
                model = models[model_index]
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_ = unwrap_model(model)
                    del models[model_index]
                elif qformer_cls is not None and isinstance(unwrap_model(model), qformer_cls):
                    qformer_ = unwrap_model(model)
                    if has_stage3c_qformer_checkpoint:
                        del models[model_index]
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
        else:
            transformer_ = Flux2Transformer2DModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="transformer",
            )
            transformer_.add_adapter(transformer_lora_config)
            if qformer is not None:
                qformer_ = unwrap_model(qformer)

        lora_state_dict = Flux2KleinPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if qformer_ is not None and has_stage3c_qformer_checkpoint:
            qformer_metadata = load_pilot_qformer_checkpoint(
                input_dir,
                qformer=qformer_,
                expected_task_id=comphoser_training.primary_task_id,
            )
            logger.info(
                "Loaded ComPhoser Q-Former checkpoint for task %s with %s query tokens",
                qformer_metadata["task_id"],
                qformer_metadata["query_count"],
            )
        elif qformer_ is not None:
            logger.warning(
                "Resuming from legacy Q-Former checkpoint layout without ComPhoser metadata under %s; "
                "falling back to Accelerate model weights.",
                input_dir,
            )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            if qformer is not None:
                models.append(unwrap_model(qformer))
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        if qformer is not None:
            models.append(qformer)
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]
    if qformer is not None:
        qformer_parameters = list(filter(lambda p: p.requires_grad, qformer.parameters()))
        params_to_optimize.append({"params": qformer_parameters, "lr": args.learning_rate})

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    if args.aspect_ratio_buckets is not None:
        buckets = parse_buckets_string(args.aspect_ratio_buckets)
    else:
        buckets = [(args.resolution, args.resolution)]
    logger.info(f"Using parsed aspect ratio buckets: {buckets}")

    # Dataset and DataLoaders creation:
    if comphoser_training.uses_prepared_pilot_dataset:
        train_dataset = PreparedPilotDataset(
            comphoser_training.dataset_roots[0],
            split="train",
            backend=args.comphoser_data_backend,
            size=args.resolution,
            repeats=args.repeats,
            center_crop=args.center_crop,
            random_flip=args.random_flip,
            buckets=buckets,
        )
        collate_train_examples = collate_prepared_pilot_examples
    else:
        train_dataset = DreamBoothDataset(
            instance_data_root=args.instance_data_dir,
            instance_prompt=args.instance_prompt,
            size=args.resolution,
            repeats=args.repeats,
            center_crop=args.center_crop,
            buckets=buckets,
        )
        collate_train_examples = collate_fn

    uses_comphoser_preprocessed_backend = bool(
        comphoser_training.uses_prepared_pilot_dataset and getattr(train_dataset, "uses_preprocessed_backend", False)
    )
    uses_comphoser_raw_backend = bool(
        comphoser_training.uses_prepared_pilot_dataset and getattr(train_dataset, "uses_raw_backend", False)
    )
    use_static_instance_prompt = (not comphoser_training.uses_prepared_pilot_dataset) and not train_dataset.custom_instance_prompts
    precompute_prompt_embeddings = (not comphoser_training.uses_prepared_pilot_dataset) and bool(
        train_dataset.custom_instance_prompts
    )
    precompute_image_latents = args.cache_latents and not uses_comphoser_preprocessed_backend
    keep_text_encoding_pipeline = uses_comphoser_raw_backend

    comphoser_checkpoint_metadata = build_pilot_checkpoint_metadata(
        args,
        train_dataset=train_dataset,
        qformer=qformer,
        comphoser_training=comphoser_training,
    )
    batch_sampler = BucketBatchSampler(train_dataset, batch_size=args.train_batch_size, drop_last=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_train_examples,
        num_workers=args.dataloader_num_workers,
    )

    def compute_text_embeddings(prompt, text_encoding_pipeline):
        with torch.no_grad():
            prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
                prompt=prompt, max_sequence_length=args.max_sequence_length
            )
        return prompt_embeds, text_ids

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    if use_static_instance_prompt:
        with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
            instance_prompt_hidden_states, instance_text_ids = compute_text_embeddings(
                args.instance_prompt, text_encoding_pipeline
            )

    validation_kwargs = None
    if not comphoser_training.uses_prepared_pilot_dataset:
        baseline_validation_prompt = args.validation_prompt if args.validation_prompt is not None else args.final_validation_prompt
    else:
        baseline_validation_prompt = None
    if baseline_validation_prompt is not None:
        validation_image = load_image(args.validation_image).convert("RGB")
        validation_kwargs = {"image": validation_image, "prompt_label": baseline_validation_prompt}
        with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
            validation_kwargs["prompt_embeds"], _text_ids = compute_text_embeddings(
                baseline_validation_prompt, text_encoding_pipeline
            )
            validation_kwargs["negative_prompt_embeds"], _text_ids = compute_text_embeddings(
                "", text_encoding_pipeline
            )

    # Init FSDP for text encoder
    if args.fsdp_text_encoder:
        fsdp_kwargs = get_fsdp_kwargs_from_accelerator(accelerator)
        text_encoder_fsdp = wrap_with_fsdp(
            model=text_encoding_pipeline.text_encoder,
            device=accelerator.device,
            offload=args.offload,
            limit_all_gathers=True,
            use_orig_params=True,
            fsdp_kwargs=fsdp_kwargs,
        )

        text_encoding_pipeline.text_encoder = text_encoder_fsdp
        dist.barrier()

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.
    if use_static_instance_prompt:
        prompt_embeds = instance_prompt_hidden_states
        text_ids = instance_text_ids

    # if cache_latents is set to True, we encode images to latents and store them.
    # Similar to pre-encoding in the case of a single instance prompt, if custom prompts are provided
    # we encode them in advance as well.
    precompute_latents = precompute_image_latents or precompute_prompt_embeddings
    if precompute_latents:
        prompt_embeds_cache = []
        text_ids_cache = []
        latents_cache = []
        cond_latents_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                if precompute_image_latents:
                    with offload_models(vae, device=accelerator.device, offload=args.offload):
                        batch["pixel_values"] = batch["pixel_values"].to(
                            accelerator.device, non_blocking=True, dtype=vae.dtype
                        )
                        latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)
                        batch["cond_pixel_values"] = batch["cond_pixel_values"].to(
                            accelerator.device, non_blocking=True, dtype=vae.dtype
                        )
                        cond_latents_cache.append(vae.encode(batch["cond_pixel_values"]).latent_dist)
                if precompute_prompt_embeddings:
                    if args.fsdp_text_encoder:
                        prompt_embeds, text_ids = compute_text_embeddings(batch["prompts"], text_encoding_pipeline)
                    else:
                        with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
                            prompt_embeds, text_ids = compute_text_embeddings(batch["prompts"], text_encoding_pipeline)
                    prompt_embeds_cache.append(prompt_embeds)
                    text_ids_cache.append(text_ids)

    # move back to cpu before deleting to ensure memory is freed see: https://github.com/huggingface/diffusers/issues/11376#issue-3008144624
    if precompute_image_latents or uses_comphoser_preprocessed_backend:
        vae = vae.to("cpu")
        del vae

    if keep_text_encoding_pipeline:
        if not args.fsdp_text_encoder:
            text_encoding_pipeline = text_encoding_pipeline.to("cpu")
    else:
        # move back to cpu before deleting to ensure memory is freed see: https://github.com/huggingface/diffusers/issues/11376#issue-3008144624
        text_encoding_pipeline = text_encoding_pipeline.to("cpu")
        del text_encoder, tokenizer
    free_memory()

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * accelerator.num_processes * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if qformer is not None:
        transformer, qformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, qformer, optimizer, train_dataloader, lr_scheduler
        )
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "dreambooth-flux2-image2img-lora"
        tracker_config = {}
        for key, value in vars(args).items():
            if isinstance(value, (bool, int, float, str, torch.Tensor)):
                tracker_config[key] = value
            elif value is None:
                tracker_config[key] = "null"
            else:
                tracker_config[key] = json.dumps(value)
        accelerator.init_trackers(tracker_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def build_validation_transformer_lora_state_dict(
        *,
        fsdp_state_dict: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        peft_kwargs = {}
        if fsdp_state_dict is not None:
            peft_kwargs["state_dict"] = fsdp_state_dict
        return {
            key: value.detach().cpu().contiguous() if isinstance(value, torch.Tensor) else value
            for key, value in get_peft_model_state_dict(unwrap_model(transformer), **peft_kwargs).items()
        }

    def build_validation_qformer_state_dict(
        *,
        fsdp_state_dict: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if qformer is None:
            return None
        source_state_dict = fsdp_state_dict if fsdp_state_dict is not None else unwrap_model(qformer).state_dict()
        return {
            key: value.detach().cpu().contiguous() if isinstance(value, torch.Tensor) else value
            for key, value in source_state_dict.items()
        }

    def run_periodic_validation(step_index: int) -> None:
        should_run_validation = (
            args.validation_steps is not None
            and args.validation_steps > 0
            and (
                (
                    not comphoser_training.uses_prepared_pilot_dataset
                    and args.validation_prompt is not None
                    and validation_kwargs is not None
                )
                or (
                    comphoser_training.uses_prepared_pilot_dataset
                    and args.comphoser_validation_mode != "off"
                )
            )
        )
        if not should_run_validation:
            return

        validation_transformer_lora_state_dict = None
        validation_qformer_state_dict = None
        if is_fsdp:
            validation_transformer_lora_state_dict = build_validation_transformer_lora_state_dict(
                fsdp_state_dict=accelerator.get_state_dict(transformer),
            )
            if qformer is not None:
                validation_qformer_state_dict = build_validation_qformer_state_dict(
                    fsdp_state_dict=accelerator.get_state_dict(qformer),
                )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if comphoser_training.uses_prepared_pilot_dataset:
                if validation_transformer_lora_state_dict is None:
                    validation_transformer_lora_state_dict = build_validation_transformer_lora_state_dict()
                validation_qformer = None
                if qformer is not None:
                    if validation_qformer_state_dict is None:
                        validation_qformer_state_dict = build_validation_qformer_state_dict()
                    validation_qformer = build_detached_validation_qformer(
                        unwrap_model(qformer),
                        state_dict=validation_qformer_state_dict,
                    )
                pipeline = build_detached_validation_pipeline(
                    pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                    transformer_lora_config=transformer_lora_config,
                    transformer_lora_state_dict=validation_transformer_lora_state_dict,
                    include_text_encoder=True,
                    enable_model_cpu_offload=True,
                    logger=logger,
                )
                run_comphoser_validation(
                    args.output_dir,
                    args=args,
                    pipelines_by_mode={args.comphoser_mode: pipeline},
                    comphoser_training=comphoser_training,
                    qformer=validation_qformer,
                    logger=logger,
                    artifact_subdir=f"periodic_validation/step_{step_index:06d}",
                )
            else:
                if validation_transformer_lora_state_dict is None:
                    validation_transformer_lora_state_dict = build_validation_transformer_lora_state_dict()
                pipeline = build_detached_validation_pipeline(
                    pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                    transformer_lora_config=transformer_lora_config,
                    transformer_lora_state_dict=validation_transformer_lora_state_dict,
                    include_text_encoder=False,
                    enable_model_cpu_offload=False,
                    logger=logger,
                )
                log_validation(
                    pipeline=pipeline,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=validation_kwargs,
                    epoch=step_index,
                    torch_dtype=weight_dtype,
                )

            del pipeline
            free_memory()
        accelerator.wait_for_everyone()

    if initial_global_step == 0:
        run_periodic_validation(0)

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        if qformer is not None:
            qformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            if qformer is not None:
                models_to_accumulate.append(qformer)
            prompts = batch["prompts"]

            with accelerator.accumulate(models_to_accumulate):
                if uses_comphoser_preprocessed_backend:
                    prompt_embeds = batch["prompt_embeds"].to(
                        device=accelerator.device,
                        non_blocking=True,
                        dtype=weight_dtype,
                    )
                    text_ids = batch["text_ids"].to(device=accelerator.device, non_blocking=True)
                elif uses_comphoser_raw_backend:
                    if args.fsdp_text_encoder:
                        prompt_embeds, text_ids = compute_text_embeddings(prompts, text_encoding_pipeline)
                    else:
                        with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
                            prompt_embeds, text_ids = compute_text_embeddings(prompts, text_encoding_pipeline)
                elif precompute_prompt_embeddings:
                    prompt_embeds = prompt_embeds_cache[step]
                    text_ids = text_ids_cache[step]
                else:
                    num_repeat_elements = len(prompts)
                    prompt_embeds = prompt_embeds.repeat(num_repeat_elements, 1, 1)
                    text_ids = text_ids.repeat(num_repeat_elements, 1, 1)

                # Convert images to latent space
                if uses_comphoser_preprocessed_backend:
                    model_input = batch["latents"].to(
                        device=accelerator.device,
                        non_blocking=True,
                        dtype=latents_bn_mean.dtype,
                    )
                    cond_model_input = batch["cond_latents"].to(
                        device=accelerator.device,
                        non_blocking=True,
                        dtype=latents_bn_mean.dtype,
                    )
                elif precompute_image_latents:
                    model_input = latents_cache[step].mode()
                    cond_model_input = cond_latents_cache[step].mode()
                else:
                    with offload_models(vae, device=accelerator.device, offload=args.offload):
                        pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=vae.dtype)
                        cond_pixel_values = batch["cond_pixel_values"].to(device=accelerator.device, dtype=vae.dtype)
                        model_input = vae.encode(pixel_values).latent_dist.mode()
                        cond_model_input = vae.encode(cond_pixel_values).latent_dist.mode()

                model_input = Flux2KleinPipeline._patchify_latents(model_input)
                model_input = (model_input - latents_bn_mean) / latents_bn_std

                cond_model_input = Flux2KleinPipeline._patchify_latents(cond_model_input)
                cond_model_input = (cond_model_input - latents_bn_mean) / latents_bn_std

                model_input_ids = Flux2KleinPipeline._prepare_latent_ids(model_input).to(device=model_input.device)
                cond_model_input_list = [cond_model_input[i].unsqueeze(0) for i in range(cond_model_input.shape[0])]
                cond_model_input_ids = Flux2KleinPipeline._prepare_image_ids(cond_model_input_list).to(
                    device=cond_model_input.device
                )
                cond_model_input_ids = cond_model_input_ids.view(
                    cond_model_input.shape[0], -1, model_input_ids.shape[-1]
                )

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                # [B, C, H, W] -> [B, H*W, C]
                # concatenate the model inputs with the cond inputs
                packed_noisy_model_input = Flux2KleinPipeline._pack_latents(noisy_model_input)
                packed_cond_model_input = Flux2KleinPipeline._pack_latents(cond_model_input)
                orig_input_shape = packed_noisy_model_input.shape
                orig_input_ids_shape = model_input_ids.shape

                # concatenate the model inputs with the cond inputs
                packed_noisy_model_input = torch.cat([packed_noisy_model_input, packed_cond_model_input], dim=1)
                model_input_ids = torch.cat([model_input_ids, cond_model_input_ids], dim=1)

                conditioning = prepare_pilot_transformer_conditioning(
                    prompt_embeds,
                    text_ids,
                    packed_cond_model_input,
                    qformer=qformer,
                    task_id=comphoser_training.primary_task_id,
                    task_strengths=(
                        resolve_pilot_batch_task_strengths(
                            batch["task_ids"],
                            batch["task_strengths"],
                            expected_task_id=comphoser_training.primary_task_id,
                        )
                        if qformer is not None
                        else None
                    ),
                )

                # handle guidance
                if unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.full([1], args.guidance_scale, device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,  # (B, image_seq_len, C)
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=conditioning.encoder_hidden_states,
                    txt_ids=conditioning.txt_ids,  # B, text_seq_len, 4
                    img_ids=model_input_ids,  # B, image_seq_len, 4
                    return_dict=False,
                )[0]
                # pruning the condition information
                model_pred = model_pred[:, : orig_input_shape[1], :]
                model_input_ids = model_input_ids[:, : orig_input_ids_shape[1], :]

                model_pred = Flux2KleinPipeline._unpack_latents_with_ids(model_pred, model_input_ids)

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - model_input

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    if qformer is not None:
                        params_to_clip = itertools.chain(params_to_clip, qformer.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process or is_fsdp:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if (
                    args.validation_steps is not None
                    and args.validation_steps > 0
                    and global_step % args.validation_steps == 0
                ):
                    run_periodic_validation(global_step)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if conditioning.query_gates is not None:
                logs["qformer_gate_mean"] = conditioning.query_gates.detach().mean().item()
                logs["qformer_added_tokens"] = float(conditioning.added_token_count)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()

    if is_fsdp:
        transformer = unwrap_model(transformer)
        state_dict = accelerator.get_state_dict(transformer)
    if accelerator.is_main_process:
        modules_to_save = {}
        if is_fsdp:
            if args.bnb_quantization_config_path is None:
                if args.upcast_before_saving:
                    state_dict = {
                        k: v.to(torch.float32) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()
                    }
                else:
                    state_dict = {
                        k: v.to(weight_dtype) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()
                    }

            transformer_lora_layers = get_peft_model_state_dict(
                transformer,
                state_dict=state_dict,
            )
            transformer_lora_layers = {
                k: v.detach().cpu().contiguous() if isinstance(v, torch.Tensor) else v
                for k, v in transformer_lora_layers.items()
            }

        else:
            transformer = unwrap_model(transformer)
            if args.bnb_quantization_config_path is None:
                if args.upcast_before_saving:
                    transformer.to(torch.float32)
                else:
                    transformer = transformer.to(weight_dtype)
            transformer_lora_layers = get_peft_model_state_dict(transformer)

        modules_to_save["transformer"] = transformer

        Flux2KleinPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )

        run_final_comphoser_export(
            args,
            comphoser_training=comphoser_training,
            qformer=qformer,
            qformer_state_dict=(
                accelerator.get_state_dict(qformer) if qformer is not None and is_fsdp else unwrap_model(qformer).state_dict()
            )
            if qformer is not None
            else None,
            comphoser_checkpoint_metadata=comphoser_checkpoint_metadata,
            weight_dtype=weight_dtype,
            unwrap_model=unwrap_model,
            logger=logger,
            run_validation=args.comphoser_mode == "lora_qformer" and args.comphoser_validation_mode == "batch",
        )

        images = []
        if comphoser_training.uses_prepared_pilot_dataset:
            should_run_final_inference = not args.skip_final_inference and (
                args.comphoser_validation_mode != "off"
                and (args.comphoser_mode != "lora_qformer" or args.comphoser_validation_mode == "single")
            )
            if should_run_final_inference:
                pipeline = Flux2KleinPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                pipeline.load_lora_weights(args.output_dir)
                pipeline.enable_model_cpu_offload()
                pipeline.set_progress_bar_config(disable=True)
                run_comphoser_validation(
                    args.output_dir,
                    args=args,
                    pipelines_by_mode={args.comphoser_mode: pipeline},
                    comphoser_training=comphoser_training,
                    qformer=unwrap_model(qformer) if qformer is not None else None,
                    logger=logger,
                    artifact_subdir="final_validation",
                )
                del pipeline
                free_memory()
        else:
            run_validation = validation_kwargs is not None and (
                (args.validation_prompt and args.num_validation_images > 0) or (args.final_validation_prompt)
            )
            should_run_final_inference = not args.skip_final_inference and run_validation
            if should_run_final_inference:
                pipeline = Flux2KleinPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                # load attention processors
                pipeline.load_lora_weights(args.output_dir)

                # run inference
                images = []
                if validation_kwargs is not None:
                    images = log_validation(
                        pipeline=pipeline,
                        args=args,
                        accelerator=accelerator,
                        pipeline_args=validation_kwargs,
                        epoch=epoch,
                        is_final_validation=True,
                        torch_dtype=weight_dtype,
                    )
                del pipeline
                free_memory()

        validation_prompt = args.validation_prompt if args.validation_prompt else args.final_validation_prompt
        save_model_card(
            (args.hub_model_id or Path(args.output_dir).name) if not args.push_to_hub else repo_id,
            images=images,
            base_model=args.pretrained_model_name_or_path,
            instance_prompt=args.instance_prompt,
            validation_prompt=validation_prompt,
            repo_folder=args.output_dir,
            fp8_training=args.do_fp8_training,
        )

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)

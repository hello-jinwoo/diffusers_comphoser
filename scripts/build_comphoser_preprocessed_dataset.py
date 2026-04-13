"""Build contract-compliant preprocessed latent caches from raw paired data."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Sequence

import torch

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))


def _load_local_preprocessing_module():
    module_path = REPO_SRC / "comphoser" / "preprocessing.py"
    module_name = "_comphoser_preprocessing_preprocessed_cli"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load preprocessing module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_PREPROCESSING = _load_local_preprocessing_module()
build_preprocessed_dataset_from_raw = _PREPROCESSING.build_preprocessed_dataset_from_raw
encode_image_latent_with_flux_vae = _PREPROCESSING.encode_image_latent_with_flux_vae
encode_prompt_latent_with_flux_text_encoder = _PREPROCESSING.encode_prompt_latent_with_flux_text_encoder
load_flux_preprocessing_models = _PREPROCESSING.load_flux_preprocessing_models


IF_EXISTS_CHOICES = ("error", "skip", "overwrite")
TORCH_DTYPE_CHOICES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "f32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct preprocessed latent caches from a raw paired dataset split.",
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Dataset split root such as data/<dataname>/train or data/<dataname>/val.",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        help="Optional explicit dataset name prefix for cache filenames.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        required=True,
        help="FLUX.2-Klein model id or local path used for latent extraction.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision.",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Optional model variant.",
    )
    parser.add_argument(
        "--torch_dtype",
        choices=tuple(TORCH_DTYPE_CHOICES.keys()),
        default="fp32",
        help="Torch dtype to use when loading the text encoder and VAE.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for latent extraction, for example cpu or cuda.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="Center-crop then resize each image to this square size before VAE encoding.",
    )
    parser.add_argument(
        "--if_exists",
        choices=IF_EXISTS_CHOICES,
        default="error",
        help="How to handle existing preprocessed cache outputs.",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="Optional limit on the number of samples to process.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum prompt sequence length for text encoding.",
    )
    return parser.parse_args(argv)


def _infer_dataset_name(dataset_root: str | Path, explicit_dataset_name: str | None) -> str | None:
    if explicit_dataset_name:
        return explicit_dataset_name

    path = Path(dataset_root).expanduser().resolve()
    if path.name in {"original", "raw", "preprocessed"}:
        path = path.parent
    if path.name in {"train", "val"}:
        return path.parent.name
    return path.name or None


def main(
    argv: Sequence[str] | None = None,
    *,
    build_preprocessed_dataset_from_raw_fn=build_preprocessed_dataset_from_raw,
    load_flux_preprocessing_models_fn=load_flux_preprocessing_models,
) -> int:
    args = parse_args(argv)
    dataset_root = Path(args.dataset_root)
    dataset_name = _infer_dataset_name(dataset_root, args.dataset_name)
    torch_dtype = TORCH_DTYPE_CHOICES[args.torch_dtype]
    tokenizer, text_encoder, vae = load_flux_preprocessing_models_fn(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch_dtype,
        device=args.device,
    )

    def image_latent_encoder(image_tensor: torch.Tensor):
        return encode_image_latent_with_flux_vae(
            image_tensor,
            vae=vae,
            device=args.device,
        )

    def prompt_latent_encoder(prompt: str):
        return encode_prompt_latent_with_flux_text_encoder(
            prompt,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=args.device,
            max_sequence_length=args.max_sequence_length,
        )

    result = build_preprocessed_dataset_from_raw_fn(
        dataset_root,
        dataset_name=dataset_name,
        image_size=args.image_size,
        if_exists=args.if_exists,
        sample_limit=args.sample_limit,
        image_latent_encoder=image_latent_encoder,
        prompt_latent_encoder=prompt_latent_encoder,
    )
    print(
        "Completed preprocessed dataset build for "
        f"{result.dataset_name}: sample_count={result.sample_count} "
        f"written={result.written_sample_count} skipped={result.skipped_sample_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

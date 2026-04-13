"""Utilities for constructing contract-compliant raw and preprocessed dataset artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from PIL.ImageOps import exif_transpose

from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

ORIGINAL_INPUT_DIR = ("original", "images", "input")
ORIGINAL_TARGET_DIR = ("original", "images", "target")
ORIGINAL_PROMPT_DIR = ("original", "prompt")
RAW_INPUT_DIR = ("raw", "images", "input")
RAW_TARGET_DIR = ("raw", "images", "target")
RAW_PROMPT_DIR = ("raw", "prompt")
PREPROCESSED_IMAGE_CACHE_INPUT_DIR = ("preprocessed", "image_latent_cache", "input")
PREPROCESSED_IMAGE_CACHE_TARGET_DIR = ("preprocessed", "image_latent_cache", "target")
PREPROCESSED_PROMPT_CACHE_DIR = ("preprocessed", "prompt_latent_cache")
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
ExistingArtifactPolicy = Literal["error", "skip", "overwrite"]
OriginalPairingMode = Literal["by_name", "sorted"]


@dataclass(frozen=True)
class OriginalPairedSample:
    dataset_root: Path
    pair_key: str
    input_image_path: Path
    target_image_path: Path
    prompt_path: Path | None = None


@dataclass(frozen=True)
class RawSamplePaths:
    input_image_path: Path
    target_image_path: Path
    prompt_path: Path

    def all_paths(self) -> tuple[Path, Path, Path]:
        return (
            self.input_image_path,
            self.target_image_path,
            self.prompt_path,
        )


@dataclass(frozen=True)
class RawDatasetBuildResult:
    dataset_root: Path
    dataset_name: str
    sample_count: int
    written_sample_count: int
    skipped_sample_count: int


@dataclass(frozen=True)
class RawPairedSample:
    dataset_root: Path
    dataset_name: str
    sample_id: str
    input_image_path: Path
    target_image_path: Path
    prompt_path: Path

    def output_paths(self) -> "PreprocessedSamplePaths":
        return build_preprocessed_sample_paths(
            self.dataset_root,
            dataset_name=self.dataset_name,
            sample_id=self.sample_id,
        )


@dataclass(frozen=True)
class PreprocessedSamplePaths:
    input_image_latent_path: Path
    target_image_latent_path: Path
    prompt_latent_path: Path

    def all_paths(self) -> tuple[Path, Path, Path]:
        return (
            self.input_image_latent_path,
            self.target_image_latent_path,
            self.prompt_latent_path,
        )


@dataclass(frozen=True)
class PreprocessedDatasetBuildResult:
    dataset_root: Path
    dataset_name: str
    sample_count: int
    written_sample_count: int
    skipped_sample_count: int


@dataclass(frozen=True)
class StagedDatasetBuildResult:
    dataset_root: Path
    dataset_name: str
    raw_result: RawDatasetBuildResult
    preprocessed_result: PreprocessedDatasetBuildResult


def infer_dataset_name(dataset_root: str | Path) -> str:
    resolved_root = resolve_dataset_root(dataset_root)
    return resolved_root.name


def resolve_dataset_root(dataset_root: str | Path) -> Path:
    resolved_root = Path(dataset_root).expanduser().resolve()
    if resolved_root.name in {"original", "raw"}:
        return resolved_root.parent
    return resolved_root


def build_raw_sample_paths(
    dataset_root: str | Path,
    *,
    dataset_name: str,
    sample_id: str,
    image_suffix: str = ".jpg",
) -> RawSamplePaths:
    dataset_root = resolve_dataset_root(dataset_root)
    normalized_suffix = _normalize_output_image_suffix(image_suffix)
    return RawSamplePaths(
        input_image_path=dataset_root / Path(*RAW_INPUT_DIR) / f"{dataset_name}_input_{sample_id}{normalized_suffix}",
        target_image_path=dataset_root / Path(*RAW_TARGET_DIR) / f"{dataset_name}_target_{sample_id}{normalized_suffix}",
        prompt_path=dataset_root / Path(*RAW_PROMPT_DIR) / f"{dataset_name}_prompt_{sample_id}.txt",
    )


def build_preprocessed_sample_paths(
    dataset_root: str | Path,
    *,
    dataset_name: str,
    sample_id: str,
) -> PreprocessedSamplePaths:
    dataset_root = resolve_dataset_root(dataset_root)
    return PreprocessedSamplePaths(
        input_image_latent_path=dataset_root
        / Path(*PREPROCESSED_IMAGE_CACHE_INPUT_DIR)
        / f"{dataset_name}_input_{sample_id}.pt",
        target_image_latent_path=dataset_root
        / Path(*PREPROCESSED_IMAGE_CACHE_TARGET_DIR)
        / f"{dataset_name}_target_{sample_id}.pt",
        prompt_latent_path=dataset_root / Path(*PREPROCESSED_PROMPT_CACHE_DIR) / f"{dataset_name}_prompt_{sample_id}.pt",
    )


def discover_original_paired_samples(
    dataset_root: str | Path,
    *,
    pairing_mode: OriginalPairingMode = "by_name",
    sample_limit: int | None = None,
    include_original_prompts: bool = False,
) -> tuple[OriginalPairedSample, ...]:
    dataset_root = resolve_dataset_root(dataset_root)
    if sample_limit is not None and sample_limit <= 0:
        raise ValueError("sample_limit must be positive when provided")
    if pairing_mode not in {"by_name", "sorted"}:
        raise ValueError(f"Unsupported pairing_mode: {pairing_mode}")

    input_root = dataset_root / Path(*ORIGINAL_INPUT_DIR)
    target_root = dataset_root / Path(*ORIGINAL_TARGET_DIR)
    prompt_root = dataset_root / Path(*ORIGINAL_PROMPT_DIR)

    for root in (input_root, target_root):
        if not root.is_dir():
            raise FileNotFoundError(f"Required original dataset directory not found: {root}")

    prompt_paths: tuple[Path, ...] | None = None
    if include_original_prompts:
        if not prompt_root.is_dir():
            raise FileNotFoundError(f"Required original prompt directory not found: {prompt_root}")
        prompt_paths = _collect_source_files(prompt_root, allowed_suffixes=(".txt",))

    if pairing_mode == "by_name":
        input_files = _collect_source_files_by_stem(input_root, allowed_suffixes=SUPPORTED_IMAGE_EXTENSIONS)
        target_files = _collect_source_files_by_stem(target_root, allowed_suffixes=SUPPORTED_IMAGE_EXTENSIONS)
        _validate_matching_original_pair_keys(
            input_files,
            target_files,
            dataset_root=dataset_root,
        )

        prompt_files: Mapping[str, Path] | None = None
        if prompt_paths is not None:
            prompt_files = _collect_source_files_by_stem(prompt_root, allowed_suffixes=(".txt",))
            _validate_matching_original_pair_keys(
                input_files,
                prompt_files,
                dataset_root=dataset_root,
                left_label="input images",
                right_label="prompts",
            )

        pair_keys = sorted(input_files.keys())
        if sample_limit is not None:
            pair_keys = pair_keys[:sample_limit]

        return tuple(
            OriginalPairedSample(
                dataset_root=dataset_root,
                pair_key=pair_key,
                input_image_path=input_files[pair_key],
                target_image_path=target_files[pair_key],
                prompt_path=None if prompt_files is None else prompt_files[pair_key],
            )
            for pair_key in pair_keys
        )

    input_paths = _collect_source_files(input_root, allowed_suffixes=SUPPORTED_IMAGE_EXTENSIONS)
    target_paths = _collect_source_files(target_root, allowed_suffixes=SUPPORTED_IMAGE_EXTENSIONS)
    if len(input_paths) != len(target_paths):
        raise ValueError(
            "Original dataset under "
            f"'{dataset_root}' does not have the same number of sorted input and target images: "
            f"input_count={len(input_paths)} target_count={len(target_paths)}"
        )

    prompt_paths_for_pairs: tuple[Path | None, ...]
    if prompt_paths is None:
        prompt_paths_for_pairs = (None,) * len(input_paths)
    else:
        if len(prompt_paths) != len(input_paths):
            raise ValueError(
                "Original dataset under "
                f"'{dataset_root}' does not have the same number of sorted prompts and image pairs: "
                f"pair_count={len(input_paths)} prompt_count={len(prompt_paths)}"
            )
        prompt_paths_for_pairs = tuple(prompt_paths)

    pair_count = len(input_paths) if sample_limit is None else min(len(input_paths), sample_limit)
    return tuple(
        OriginalPairedSample(
            dataset_root=dataset_root,
            pair_key=f"{index + 1:06d}",
            input_image_path=input_paths[index],
            target_image_path=target_paths[index],
            prompt_path=prompt_paths_for_pairs[index],
        )
        for index in range(pair_count)
    )


def discover_raw_paired_samples(
    dataset_root: str | Path,
    *,
    dataset_name: str | None = None,
    sample_limit: int | None = None,
) -> tuple[RawPairedSample, ...]:
    dataset_root = resolve_dataset_root(dataset_root)
    dataset_name = dataset_name or dataset_root.name
    if sample_limit is not None and sample_limit <= 0:
        raise ValueError("sample_limit must be positive when provided")

    input_root = dataset_root / Path(*RAW_INPUT_DIR)
    target_root = dataset_root / Path(*RAW_TARGET_DIR)
    prompt_root = dataset_root / Path(*RAW_PROMPT_DIR)

    for root in (input_root, target_root, prompt_root):
        if not root.is_dir():
            raise FileNotFoundError(f"Required raw dataset directory not found: {root}")

    input_files = _collect_named_files(input_root, prefix=f"{dataset_name}_input_", allowed_suffixes=SUPPORTED_IMAGE_EXTENSIONS)
    target_files = _collect_named_files(
        target_root,
        prefix=f"{dataset_name}_target_",
        allowed_suffixes=SUPPORTED_IMAGE_EXTENSIONS,
    )
    prompt_files = _collect_named_files(prompt_root, prefix=f"{dataset_name}_prompt_", allowed_suffixes=(".txt",))

    _validate_sample_triplets(input_files, target_files, prompt_files, dataset_root=dataset_root, dataset_name=dataset_name)

    sample_ids = sorted(input_files.keys())
    if sample_limit is not None:
        sample_ids = sample_ids[:sample_limit]

    return tuple(
        RawPairedSample(
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            sample_id=sample_id,
            input_image_path=input_files[sample_id],
            target_image_path=target_files[sample_id],
            prompt_path=prompt_files[sample_id],
        )
        for sample_id in sample_ids
    )


def preprocess_image_for_contract(image: Image.Image, *, size: int = 1024) -> Image.Image:
    if size <= 0:
        raise ValueError("size must be positive")

    image = exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # The dataset contract is explicit here: center-crop first, then resize to 1024x1024.
    return ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def load_and_preprocess_contract_image(image_path: str | Path, *, size: int = 1024) -> torch.Tensor:
    with Image.open(image_path) as image:
        processed = preprocess_image_for_contract(image, size=size)
    return pil_image_to_normalized_tensor(processed)


def pil_image_to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    image_array = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
    return (tensor / 127.5) - 1.0


def read_prompt_text(prompt_path: str | Path) -> str:
    text = Path(prompt_path).read_text(encoding="utf-8")
    return text.rstrip("\r\n")


def encode_image_latent_with_flux_vae(
    image_tensor: torch.Tensor,
    *,
    vae,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    if image_tensor.ndim != 3:
        raise ValueError(f"Expected image_tensor with shape [channels, height, width], received ndim={image_tensor.ndim}")

    device = torch.device(device or _module_device(vae))
    dtype = _module_dtype(vae)

    with torch.no_grad():
        batch = image_tensor.unsqueeze(0).to(device=device, dtype=dtype)
        image_latent = vae.encode(batch).latent_dist.mode()[0]
    return image_latent.detach().cpu()


def encode_prompt_latent_with_flux_text_encoder(
    prompt: str,
    *,
    tokenizer,
    text_encoder,
    device: str | torch.device | None = None,
    max_sequence_length: int = 512,
    text_encoder_out_layers: Sequence[int] = (9, 18, 27),
) -> dict[str, torch.Tensor]:
    device = torch.device(device or _module_device(text_encoder))
    dtype = _module_dtype(text_encoder)

    with torch.no_grad():
        prompt_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            dtype=dtype,
            max_sequence_length=max_sequence_length,
            hidden_states_layers=list(text_encoder_out_layers),
        )
        text_ids = Flux2KleinPipeline._prepare_text_ids(prompt_embeds).to(device=device)

    return {
        "prompt_embeds": prompt_embeds[0].detach().cpu(),
        "text_ids": text_ids[0].detach().cpu(),
    }


def load_flux_preprocessing_models(
    *,
    pretrained_model_name_or_path: str,
    revision: str | None = None,
    variant: str | None = None,
    torch_dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
):
    from diffusers import AutoencoderKLFlux2
    from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
    )
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
        variant=variant,
        torch_dtype=torch_dtype,
    )
    vae = AutoencoderKLFlux2.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        revision=revision,
        variant=variant,
        torch_dtype=torch_dtype,
    )

    device = torch.device(device)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.eval().to(device=device, dtype=torch_dtype)
    vae.eval().to(device=device, dtype=torch_dtype)

    return tokenizer, text_encoder, vae


def build_raw_dataset_from_original(
    dataset_root: str | Path,
    *,
    prompt_builder: Callable[[OriginalPairedSample, str, int], str],
    dataset_name: str | None = None,
    pairing_mode: OriginalPairingMode = "by_name",
    image_size: int = 1024,
    output_image_suffix: str = ".jpg",
    if_exists: ExistingArtifactPolicy = "error",
    sample_limit: int | None = None,
    sample_id_width: int = 6,
    include_original_prompts: bool = False,
) -> RawDatasetBuildResult:
    if if_exists not in {"error", "skip", "overwrite"}:
        raise ValueError(f"Unsupported if_exists policy: {if_exists}")
    if sample_id_width <= 0:
        raise ValueError("sample_id_width must be positive")

    normalized_suffix = _normalize_output_image_suffix(output_image_suffix)
    samples = discover_original_paired_samples(
        dataset_root,
        pairing_mode=pairing_mode,
        sample_limit=sample_limit,
        include_original_prompts=include_original_prompts,
    )
    resolved_root = resolve_dataset_root(dataset_root)
    resolved_name = dataset_name or resolved_root.name

    written_sample_count = 0
    skipped_sample_count = 0
    for sample_index, sample in enumerate(samples, start=1):
        sample_id = f"{sample_index:0{sample_id_width}d}"
        output_paths = build_raw_sample_paths(
            resolved_root,
            dataset_name=resolved_name,
            sample_id=sample_id,
            image_suffix=normalized_suffix,
        )
        existing_paths = tuple(path for path in output_paths.all_paths() if path.exists())
        if existing_paths:
            if if_exists == "overwrite":
                pass
            elif if_exists == "skip":
                if len(existing_paths) != len(output_paths.all_paths()):
                    existing_text = ", ".join(str(path) for path in existing_paths)
                    raise FileExistsError(
                        "Refusing to skip a partially built raw sample. "
                        f"Existing paths for sample '{sample_id}': {existing_text}"
                    )
                skipped_sample_count += 1
                continue
            else:
                existing_text = ", ".join(str(path) for path in existing_paths)
                raise FileExistsError(f"Raw sample already exists for sample '{sample_id}': {existing_text}")

        for directory in (
            output_paths.input_image_path.parent,
            output_paths.target_image_path.parent,
            output_paths.prompt_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        _save_contract_image(sample.input_image_path, output_paths.input_image_path, size=image_size)
        _save_contract_image(sample.target_image_path, output_paths.target_image_path, size=image_size)

        prompt_text = prompt_builder(sample, sample_id, sample_index)
        normalized_prompt = str(prompt_text).rstrip("\r\n")
        output_paths.prompt_path.write_text(f"{normalized_prompt}\n", encoding="utf-8")
        written_sample_count += 1

    return RawDatasetBuildResult(
        dataset_root=resolved_root,
        dataset_name=resolved_name,
        sample_count=len(samples),
        written_sample_count=written_sample_count,
        skipped_sample_count=skipped_sample_count,
    )


def build_raw_images_from_original(
    dataset_root: str | Path,
    *,
    dataset_name: str | None = None,
    pairing_mode: OriginalPairingMode = "by_name",
    image_size: int = 1024,
    output_image_suffix: str = ".jpg",
    if_exists: ExistingArtifactPolicy = "error",
    sample_limit: int | None = None,
    sample_id_width: int = 6,
) -> RawDatasetBuildResult:
    if if_exists not in {"error", "skip", "overwrite"}:
        raise ValueError(f"Unsupported if_exists policy: {if_exists}")
    if sample_id_width <= 0:
        raise ValueError("sample_id_width must be positive")

    normalized_suffix = _normalize_output_image_suffix(output_image_suffix)
    samples = discover_original_paired_samples(
        dataset_root,
        pairing_mode=pairing_mode,
        sample_limit=sample_limit,
        include_original_prompts=False,
    )
    resolved_root = resolve_dataset_root(dataset_root)
    resolved_name = dataset_name or resolved_root.name

    written_sample_count = 0
    skipped_sample_count = 0
    for sample_index, sample in enumerate(samples, start=1):
        sample_id = f"{sample_index:0{sample_id_width}d}"
        output_paths = build_raw_sample_paths(
            resolved_root,
            dataset_name=resolved_name,
            sample_id=sample_id,
            image_suffix=normalized_suffix,
        )
        image_paths = (
            output_paths.input_image_path,
            output_paths.target_image_path,
        )
        existing_paths = tuple(path for path in image_paths if path.exists())
        if existing_paths:
            if if_exists == "overwrite":
                pass
            elif if_exists == "skip":
                if len(existing_paths) != len(image_paths):
                    existing_text = ", ".join(str(path) for path in existing_paths)
                    raise FileExistsError(
                        "Refusing to skip a partially built raw image sample. "
                        f"Existing paths for sample '{sample_id}': {existing_text}"
                    )
                skipped_sample_count += 1
                continue
            else:
                existing_text = ", ".join(str(path) for path in existing_paths)
                raise FileExistsError(f"Raw image sample already exists for sample '{sample_id}': {existing_text}")

        for directory in (
            output_paths.input_image_path.parent,
            output_paths.target_image_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        _save_contract_image(sample.input_image_path, output_paths.input_image_path, size=image_size)
        _save_contract_image(sample.target_image_path, output_paths.target_image_path, size=image_size)
        written_sample_count += 1

    return RawDatasetBuildResult(
        dataset_root=resolved_root,
        dataset_name=resolved_name,
        sample_count=len(samples),
        written_sample_count=written_sample_count,
        skipped_sample_count=skipped_sample_count,
    )


def build_preprocessed_dataset_from_raw(
    dataset_root: str | Path,
    *,
    image_latent_encoder: Callable[[torch.Tensor], torch.Tensor],
    prompt_latent_encoder: Callable[[str], Any],
    dataset_name: str | None = None,
    image_size: int = 1024,
    if_exists: ExistingArtifactPolicy = "error",
    sample_limit: int | None = None,
) -> PreprocessedDatasetBuildResult:
    if if_exists not in {"error", "skip", "overwrite"}:
        raise ValueError(f"Unsupported if_exists policy: {if_exists}")

    samples = discover_raw_paired_samples(
        dataset_root,
        dataset_name=dataset_name,
        sample_limit=sample_limit,
    )
    resolved_root = resolve_dataset_root(dataset_root)
    resolved_name = dataset_name or resolved_root.name

    written_sample_count = 0
    skipped_sample_count = 0
    for sample in samples:
        output_paths = sample.output_paths()
        existing_paths = tuple(path for path in output_paths.all_paths() if path.exists())
        if existing_paths:
            if if_exists == "overwrite":
                pass
            elif if_exists == "skip":
                if len(existing_paths) != len(output_paths.all_paths()):
                    existing_text = ", ".join(str(path) for path in existing_paths)
                    raise FileExistsError(
                        "Refusing to skip a partially built sample cache. "
                        f"Existing paths for sample '{sample.sample_id}': {existing_text}"
                    )
                skipped_sample_count += 1
                continue
            else:
                existing_text = ", ".join(str(path) for path in existing_paths)
                raise FileExistsError(
                    f"Preprocessed cache already exists for sample '{sample.sample_id}': {existing_text}"
                )

        for directory in (
            output_paths.input_image_latent_path.parent,
            output_paths.target_image_latent_path.parent,
            output_paths.prompt_latent_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        input_image_tensor = load_and_preprocess_contract_image(sample.input_image_path, size=image_size)
        target_image_tensor = load_and_preprocess_contract_image(sample.target_image_path, size=image_size)
        prompt_text = read_prompt_text(sample.prompt_path)

        torch.save(image_latent_encoder(input_image_tensor), output_paths.input_image_latent_path)
        torch.save(image_latent_encoder(target_image_tensor), output_paths.target_image_latent_path)
        torch.save(prompt_latent_encoder(prompt_text), output_paths.prompt_latent_path)
        written_sample_count += 1

    return PreprocessedDatasetBuildResult(
        dataset_root=resolved_root,
        dataset_name=resolved_name,
        sample_count=len(samples),
        written_sample_count=written_sample_count,
        skipped_sample_count=skipped_sample_count,
    )


def build_dataset_from_original(
    dataset_root: str | Path,
    *,
    prompt_builder: Callable[[OriginalPairedSample, str, int], str],
    dataset_name: str | None = None,
    pairing_mode: OriginalPairingMode = "by_name",
    image_size: int = 1024,
    output_image_suffix: str = ".jpg",
    raw_if_exists: ExistingArtifactPolicy = "error",
    preprocessed_if_exists: ExistingArtifactPolicy = "error",
    sample_limit: int | None = None,
    sample_id_width: int = 6,
    include_original_prompts: bool = False,
    image_latent_encoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
    prompt_latent_encoder: Callable[[str], Any] | None = None,
    pretrained_model_name_or_path: str | None = None,
    revision: str | None = None,
    variant: str | None = None,
    torch_dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    max_sequence_length: int = 512,
) -> StagedDatasetBuildResult:
    raw_result = build_raw_dataset_from_original(
        dataset_root,
        prompt_builder=prompt_builder,
        dataset_name=dataset_name,
        pairing_mode=pairing_mode,
        image_size=image_size,
        output_image_suffix=output_image_suffix,
        if_exists=raw_if_exists,
        sample_limit=sample_limit,
        sample_id_width=sample_id_width,
        include_original_prompts=include_original_prompts,
    )

    if (image_latent_encoder is None) != (prompt_latent_encoder is None):
        raise ValueError(
            "image_latent_encoder and prompt_latent_encoder must both be provided or both omitted"
        )

    if image_latent_encoder is None or prompt_latent_encoder is None:
        if not pretrained_model_name_or_path:
            raise ValueError(
                "pretrained_model_name_or_path is required when explicit latent encoders are not provided"
            )

        tokenizer, text_encoder, vae = load_flux_preprocessing_models(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            revision=revision,
            variant=variant,
            torch_dtype=torch_dtype,
            device=device,
        )
        image_latent_encoder = lambda image_tensor: encode_image_latent_with_flux_vae(
            image_tensor,
            vae=vae,
            device=device,
        )
        prompt_latent_encoder = lambda prompt: encode_prompt_latent_with_flux_text_encoder(
            prompt,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=device,
            max_sequence_length=max_sequence_length,
        )

    preprocessed_result = build_preprocessed_dataset_from_raw(
        dataset_root,
        dataset_name=dataset_name,
        image_size=image_size,
        if_exists=preprocessed_if_exists,
        sample_limit=sample_limit,
        image_latent_encoder=image_latent_encoder,
        prompt_latent_encoder=prompt_latent_encoder,
    )

    return StagedDatasetBuildResult(
        dataset_root=raw_result.dataset_root,
        dataset_name=raw_result.dataset_name,
        raw_result=raw_result,
        preprocessed_result=preprocessed_result,
    )


def _collect_named_files(
    directory: Path,
    *,
    prefix: str,
    allowed_suffixes: Sequence[str],
) -> dict[str, Path]:
    files_by_id: dict[str, Path] = {}
    normalized_suffixes = tuple(suffix.lower() for suffix in allowed_suffixes)
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in normalized_suffixes:
            continue
        if not path.stem.startswith(prefix):
            continue
        sample_id = path.stem[len(prefix) :]
        if not sample_id:
            raise ValueError(f"Could not extract a sample id from file '{path.name}'")
        if sample_id in files_by_id:
            raise ValueError(f"Duplicate files detected for sample id '{sample_id}' in {directory}")
        files_by_id[sample_id] = path
    return files_by_id


def _collect_source_files(
    directory: Path,
    *,
    allowed_suffixes: Sequence[str],
) -> tuple[Path, ...]:
    normalized_suffixes = tuple(suffix.lower() for suffix in allowed_suffixes)
    return tuple(
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in normalized_suffixes
    )


def _collect_source_files_by_stem(
    directory: Path,
    *,
    allowed_suffixes: Sequence[str],
) -> dict[str, Path]:
    files_by_stem: dict[str, Path] = {}
    for path in _collect_source_files(directory, allowed_suffixes=allowed_suffixes):
        if path.stem in files_by_stem:
            raise ValueError(f"Duplicate files detected for pair key '{path.stem}' in {directory}")
        files_by_stem[path.stem] = path
    return files_by_stem


def _normalize_output_image_suffix(image_suffix: str) -> str:
    normalized_suffix = image_suffix.lower()
    if not normalized_suffix.startswith("."):
        normalized_suffix = f".{normalized_suffix}"
    if normalized_suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported output image suffix: {image_suffix}")
    return normalized_suffix


def _save_contract_image(source_path: Path, destination_path: Path, *, size: int) -> None:
    with Image.open(source_path) as image:
        processed = preprocess_image_for_contract(image, size=size)
    processed.save(destination_path, format=_pil_image_format_from_suffix(destination_path.suffix))


def _pil_image_format_from_suffix(image_suffix: str) -> str:
    normalized_suffix = _normalize_output_image_suffix(image_suffix)
    if normalized_suffix in {".jpg", ".jpeg"}:
        return "JPEG"
    if normalized_suffix == ".png":
        return "PNG"
    if normalized_suffix == ".webp":
        return "WEBP"
    if normalized_suffix == ".bmp":
        return "BMP"
    raise ValueError(f"Unsupported output image suffix: {image_suffix}")


def _validate_matching_original_pair_keys(
    left_files: Mapping[str, Path],
    right_files: Mapping[str, Path],
    *,
    dataset_root: Path,
    left_label: str = "input images",
    right_label: str = "target images",
) -> None:
    left_keys = set(left_files)
    right_keys = set(right_files)
    missing_left = sorted(right_keys - left_keys)
    missing_right = sorted(left_keys - right_keys)

    if not missing_left and not missing_right:
        return

    messages: list[str] = []
    if missing_left:
        messages.append(f"missing {left_label} for pair keys: {', '.join(missing_left)}")
    if missing_right:
        messages.append(f"missing {right_label} for pair keys: {', '.join(missing_right)}")
    raise ValueError(
        f"Original dataset under '{dataset_root}' does not form complete paired samples: {'; '.join(messages)}"
    )


def _validate_sample_triplets(
    input_files: Mapping[str, Path],
    target_files: Mapping[str, Path],
    prompt_files: Mapping[str, Path],
    *,
    dataset_root: Path,
    dataset_name: str,
) -> None:
    input_ids = set(input_files)
    target_ids = set(target_files)
    prompt_ids = set(prompt_files)
    all_ids = input_ids | target_ids | prompt_ids

    missing_messages: list[str] = []
    for label, ids in (
        ("input images", input_ids),
        ("target images", target_ids),
        ("prompts", prompt_ids),
    ):
        missing = sorted(all_ids - ids)
        if missing:
            missing_messages.append(f"missing {label} for sample ids: {', '.join(missing)}")

    if missing_messages:
        message = "; ".join(missing_messages)
        raise ValueError(
            f"Raw dataset '{dataset_name}' under '{dataset_root}' does not form complete input/target/prompt triplets: {message}"
        )


def _module_device(module: Any) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is None:
        return torch.device("cpu")
    return parameter.device


def _module_dtype(module: Any) -> torch.dtype:
    parameter = next(module.parameters(), None)
    if parameter is None:
        return torch.float32
    return parameter.dtype

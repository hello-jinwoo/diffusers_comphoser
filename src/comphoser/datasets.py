"""Prepared-dataset bridge for the first ComPhoser Q-Former pilot."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch import Tensor
from torch.utils.data import Dataset

from .controls import get_task_spec_for_dataset_id
from .preprocessing import build_preprocessed_sample_paths, discover_raw_paired_samples, read_prompt_text

COMPHOSER_DATA_BACKENDS = ("preprocessed", "raw")
PREPARED_PRIMITIVE_FORMAT_VERSION = "comphoser-prepared-primitive-v1"
PREPARED_PRIMITIVE_METADATA_FILENAME = "comphoser_prepared_dataset.json"
PREPARED_RECORD_SOURCE_MANIFEST = "manifest"
PREPARED_RECORD_SOURCE_DERIVED_CONTRACT = "derived_contract_split"


@dataclass(frozen=True)
class PreparedPilotDatasetMetadata:
    dataset_root: Path
    raw_root: Path | None
    dataset_id: str
    format_version: str
    image_column: str
    cond_image_column: str
    caption_column: str
    task_ids_column: str
    task_strengths_column: str
    prepared_splits: tuple[str, ...]
    record_source: str = PREPARED_RECORD_SOURCE_MANIFEST

    def manifest_path_for_split(self, split: str) -> Path:
        if split not in self.prepared_splits:
            supported = ", ".join(self.prepared_splits)
            raise ValueError(f"Unsupported prepared split '{split}'. Available prepared splits: {supported}")
        if self.record_source == PREPARED_RECORD_SOURCE_DERIVED_CONTRACT:
            return self.dataset_root / split
        return self.dataset_root / "manifests" / f"{split}.jsonl"

    def resolve_asset_path(self, relative_path: str) -> Path:
        return _resolve_prepared_or_raw_path(self.dataset_root, self.raw_root, relative_path)


@dataclass(frozen=True)
class PreparedPilotRecord:
    dataset_id: str
    split: str
    sample_id: str
    prompt: str
    source_prompt: str | None
    mode: str
    primitive_family: str | None
    image_relpath: str
    cond_image_relpath: str
    image_path: Path
    cond_image_path: Path
    task_ids: tuple[str, ...]
    task_strengths: tuple[float, ...]
    source_sample_id: str | None
    source_task: str | None
    weight: float
    metadata: Mapping[str, Any]


class PreparedPilotDataset(Dataset):
    """Trainer-facing dataset for the prepared single-task pilot manifests."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: str,
        backend: str = "raw",
        size: int | tuple[int, int] = 1024,
        repeats: int = 1,
        center_crop: bool = False,
        random_flip: bool = False,
        buckets: Sequence[tuple[int, int]] | None = None,
        sample_limit: int | None = None,
    ) -> None:
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        if sample_limit is not None and sample_limit <= 0:
            raise ValueError("sample_limit must be positive when provided")
        if backend not in COMPHOSER_DATA_BACKENDS:
            supported = ", ".join(COMPHOSER_DATA_BACKENDS)
            raise ValueError(f"Unsupported PreparedPilotDataset backend '{backend}'. Expected one of: {supported}")

        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split = split
        self.backend = backend
        self.size = _normalize_size(size)
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.metadata = load_prepared_pilot_dataset_metadata(self.dataset_root)
        self.buckets = _normalize_buckets(buckets, default_size=self.size)

        records = list(load_prepared_pilot_records(self.dataset_root, split=split))
        if sample_limit is not None:
            records = records[:sample_limit]
        self.records = tuple(records)
        self._repeated_records = tuple(record for record in self.records for _ in range(repeats))
        self.bucket_indices = tuple(self._bucket_index_for_record(record) for record in self._repeated_records)
        self.custom_instance_prompts = [record.prompt for record in self._repeated_records]

    def __len__(self) -> int:
        return len(self._repeated_records)

    @property
    def uses_preprocessed_backend(self) -> bool:
        return self.backend == "preprocessed"

    @property
    def uses_raw_backend(self) -> bool:
        return self.backend == "raw"

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._repeated_records[index]
        bucket_idx = self.bucket_indices[index]
        example = {
            "backend": self.backend,
            "instance_prompt": record.prompt,
            "bucket_idx": bucket_idx,
            "sample_id": record.sample_id,
            "dataset_id": record.dataset_id,
            "mode": record.mode,
            "task_ids": record.task_ids,
            "task_strengths": record.task_strengths,
            "source_prompt": record.source_prompt,
            "source_sample_id": record.source_sample_id,
            "source_task": record.source_task,
            "primitive_family": record.primitive_family,
            "weight": record.weight,
        }
        if self.uses_preprocessed_backend:
            cache_paths = _resolve_preprocessed_cache_paths(record, dataset_root=self.dataset_root)
            prompt_cache = _load_prompt_cache_payload(cache_paths["prompt"])
            example.update(
                {
                    "instance_latents": _load_cached_latent_tensor(cache_paths["target"], context=record.sample_id),
                    "cond_latents": _load_cached_latent_tensor(cache_paths["input"], context=record.sample_id),
                    "prompt_embeds": prompt_cache["prompt_embeds"],
                    "text_ids": prompt_cache["text_ids"],
                }
            )
            return example

        bucket_size = self.buckets[bucket_idx]
        instance_image = _load_rgb_image(record.image_path)
        cond_image = _load_rgb_image(record.cond_image_path)
        instance_tensor, cond_tensor = _paired_transform(
            instance_image,
            cond_image,
            size=bucket_size,
            center_crop=self.center_crop,
            random_flip=self.random_flip,
        )
        example.update(
            {
                "instance_images": instance_tensor,
                "cond_images": cond_tensor,
            }
        )
        return example

    def _bucket_index_for_record(self, record: PreparedPilotRecord) -> int:
        with Image.open(record.image_path) as image:
            width, height = image.size
        return find_nearest_bucket(height, width, self.buckets)


def load_prepared_pilot_dataset_metadata(dataset_root: str | Path) -> PreparedPilotDatasetMetadata:
    dataset_root = Path(dataset_root).expanduser().resolve()
    metadata_path = dataset_root / PREPARED_PRIMITIVE_METADATA_FILENAME
    if not metadata_path.is_file():
        return _derive_contract_prepared_metadata(dataset_root)

    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    format_version = payload.get("format_version")
    if format_version != PREPARED_PRIMITIVE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported prepared dataset format '{format_version}'. "
            f"Expected '{PREPARED_PRIMITIVE_FORMAT_VERSION}'."
        )

    required_fields = (
        "dataset_id",
        "image_column",
        "cond_image_column",
        "caption_column",
        "task_ids_column",
        "task_strengths_column",
        "prepared_splits",
    )
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"Prepared dataset metadata is missing required fields: {missing}")

    prepared_splits = payload["prepared_splits"]
    if (
        not isinstance(prepared_splits, list)
        or not prepared_splits
        or not all(isinstance(split, str) and split for split in prepared_splits)
    ):
        raise ValueError("prepared_splits must be a non-empty list of strings")

    return PreparedPilotDatasetMetadata(
        dataset_root=dataset_root,
        raw_root=_discover_raw_root(dataset_root),
        dataset_id=_require_string(payload, "dataset_id", context=str(metadata_path)),
        format_version=format_version,
        image_column=_require_string(payload, "image_column", context=str(metadata_path)),
        cond_image_column=_require_string(payload, "cond_image_column", context=str(metadata_path)),
        caption_column=_require_string(payload, "caption_column", context=str(metadata_path)),
        task_ids_column=_require_string(payload, "task_ids_column", context=str(metadata_path)),
        task_strengths_column=_require_string(payload, "task_strengths_column", context=str(metadata_path)),
        prepared_splits=tuple(prepared_splits),
        record_source=PREPARED_RECORD_SOURCE_MANIFEST,
    )


def load_prepared_pilot_records(
    dataset_root: str | Path,
    *,
    split: str,
) -> tuple[PreparedPilotRecord, ...]:
    metadata = load_prepared_pilot_dataset_metadata(dataset_root)
    if metadata.record_source == PREPARED_RECORD_SOURCE_DERIVED_CONTRACT:
        return _load_derived_contract_records(metadata, split=split)

    manifest_path = metadata.manifest_path_for_split(split)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared manifest file not found: {manifest_path}")

    records: list[PreparedPilotRecord] = []
    seen_sample_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            record = _coerce_prepared_record(
                payload,
                metadata=metadata,
                split=split,
                line_number=line_number,
            )
            if record.sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate sample_id '{record.sample_id}' found in {manifest_path}")
            seen_sample_ids.add(record.sample_id)
            records.append(record)

    return tuple(records)


def _derive_contract_prepared_metadata(dataset_root: Path) -> PreparedPilotDatasetMetadata:
    prepared_splits = tuple(
        split_name
        for split_name in ("train", "val")
        if _is_contract_split_root(dataset_root / split_name)
    )
    if not prepared_splits:
        metadata_path = dataset_root / PREPARED_PRIMITIVE_METADATA_FILENAME
        raise FileNotFoundError(
            f"Prepared dataset metadata file not found: {metadata_path}. "
            f"No contract dataset splits were discovered under {dataset_root}."
        )

    return PreparedPilotDatasetMetadata(
        dataset_root=dataset_root,
        raw_root=None,
        dataset_id=dataset_root.name,
        format_version=PREPARED_PRIMITIVE_FORMAT_VERSION,
        image_column="edited_image",
        cond_image_column="input_image",
        caption_column="prompt",
        task_ids_column="task_ids",
        task_strengths_column="task_strengths",
        prepared_splits=prepared_splits,
        record_source=PREPARED_RECORD_SOURCE_DERIVED_CONTRACT,
    )


def _load_derived_contract_records(
    metadata: PreparedPilotDatasetMetadata,
    *,
    split: str,
) -> tuple[PreparedPilotRecord, ...]:
    split_root = metadata.manifest_path_for_split(split)
    if not _is_contract_split_root(split_root):
        raise FileNotFoundError(f"Contract dataset split root not found or incomplete: {split_root}")

    runtime_spec = _resolve_contract_runtime_spec(metadata.dataset_root)
    source_records = discover_raw_paired_samples(split_root, dataset_name=metadata.dataset_id)
    records: list[PreparedPilotRecord] = []
    seen_sample_ids: set[str] = set()
    for record in _build_derived_contract_records(
        source_records,
        dataset_root=metadata.dataset_root,
        split=split,
        dataset_id=metadata.dataset_id,
        task_id=runtime_spec["task_id"],
        primitive_family=runtime_spec["primitive_family"],
    ):
        if record.sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate sample_id '{record.sample_id}' derived for contract split '{split_root}'")
        seen_sample_ids.add(record.sample_id)
        records.append(record)

    return tuple(records)


def collate_prepared_pilot_examples(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not examples:
        raise ValueError("Cannot collate an empty batch of prepared pilot examples")
    backend = str(examples[0]["backend"])
    if any(str(example["backend"]) != backend for example in examples):
        raise ValueError("Prepared pilot batches cannot mix different backends in one collate call")

    batch = {
        "backend": backend,
        "prompts": [str(example["instance_prompt"]) for example in examples],
        "sample_ids": tuple(str(example["sample_id"]) for example in examples),
        "dataset_ids": tuple(str(example["dataset_id"]) for example in examples),
        "modes": tuple(str(example["mode"]) for example in examples),
        "task_ids": tuple(tuple(example["task_ids"]) for example in examples),
        "task_strengths": tuple(tuple(example["task_strengths"]) for example in examples),
        "source_prompts": tuple(example.get("source_prompt") for example in examples),
        "source_sample_ids": tuple(example.get("source_sample_id") for example in examples),
        "source_tasks": tuple(example.get("source_task") for example in examples),
        "primitive_families": tuple(example.get("primitive_family") for example in examples),
        "weights": torch.tensor([float(example.get("weight", 1.0)) for example in examples], dtype=torch.float32),
    }
    if backend == "preprocessed":
        batch.update(
            {
                "latents": torch.stack([example["instance_latents"] for example in examples])
                .to(memory_format=torch.contiguous_format)
                .float(),
                "cond_latents": torch.stack([example["cond_latents"] for example in examples])
                .to(memory_format=torch.contiguous_format)
                .float(),
                "prompt_embeds": torch.stack([example["prompt_embeds"] for example in examples])
                .to(memory_format=torch.contiguous_format)
                .float(),
                "text_ids": torch.stack([example["text_ids"] for example in examples]).contiguous().long(),
            }
        )
        return batch

    pixel_values = torch.stack([example["instance_images"] for example in examples])
    cond_pixel_values = torch.stack([example["cond_images"] for example in examples])
    batch.update(
        {
            "pixel_values": pixel_values.to(memory_format=torch.contiguous_format).float(),
            "cond_pixel_values": cond_pixel_values.to(memory_format=torch.contiguous_format).float(),
        }
    )
    return batch


def find_nearest_bucket(h: int, w: int, bucket_options: Sequence[tuple[int, int]]) -> int:
    min_metric = float("inf")
    best_bucket_idx = 0
    for bucket_idx, (bucket_h, bucket_w) in enumerate(bucket_options):
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            best_bucket_idx = bucket_idx
    return best_bucket_idx


def _coerce_prepared_record(
    payload: Mapping[str, Any],
    *,
    metadata: PreparedPilotDatasetMetadata,
    split: str,
    line_number: int,
) -> PreparedPilotRecord:
    context = f"{metadata.manifest_path_for_split(split)}:{line_number}"
    image_relpath = _require_string(payload, metadata.image_column, context=context)
    cond_image_relpath = _require_string(payload, metadata.cond_image_column, context=context)
    prompt = _require_string(payload, metadata.caption_column, context=context)
    sample_id = _require_string(payload, "sample_id", context=context)
    dataset_id = _require_string(payload, "dataset_id", context=context)
    mode = _require_string(payload, "mode", context=context)
    if dataset_id != metadata.dataset_id:
        raise ValueError(
            f"{context} dataset_id '{dataset_id}' does not match prepared metadata dataset_id '{metadata.dataset_id}'"
        )
    if mode not in {"task", "identity"}:
        raise ValueError(f"{context} mode must be 'task' or 'identity', received '{mode}'")

    task_ids = payload.get(metadata.task_ids_column, [])
    task_strengths = payload.get(metadata.task_strengths_column, [])
    if not isinstance(task_ids, list) or not all(isinstance(task_id, str) and task_id for task_id in task_ids):
        raise ValueError(f"{context} must provide '{metadata.task_ids_column}' as a list of non-empty strings")
    if not isinstance(task_strengths, list):
        raise ValueError(f"{context} must provide '{metadata.task_strengths_column}' as a list of floats")
    if len(task_ids) != len(task_strengths):
        raise ValueError(
            f"{context} has mismatched '{metadata.task_ids_column}' and '{metadata.task_strengths_column}' lengths"
        )

    normalized_strengths = tuple(float(strength) for strength in task_strengths)
    if any(strength < 0.0 or strength > 1.0 for strength in normalized_strengths):
        raise ValueError(f"{context} task strengths must stay within [0.0, 1.0]")
    if payload.get("split") is not None and payload["split"] != split:
        raise ValueError(f"{context} split field '{payload['split']}' does not match requested split '{split}'")

    metadata_json = payload.get("metadata_json")
    if metadata_json is None:
        resolved_metadata: Mapping[str, Any] = {}
    elif isinstance(metadata_json, str):
        resolved_metadata = json.loads(metadata_json)
    elif isinstance(metadata_json, Mapping):
        resolved_metadata = metadata_json
    else:
        raise ValueError(f"{context} has unsupported metadata_json type: {type(metadata_json).__name__}")

    weight = float(payload.get("weight", 1.0))
    if weight <= 0.0:
        raise ValueError(f"{context} weight must be positive")

    return PreparedPilotRecord(
        dataset_id=dataset_id,
        split=split,
        sample_id=sample_id,
        prompt=prompt,
        source_prompt=payload.get("source_prompt"),
        mode=mode,
        primitive_family=payload.get("primitive_family"),
        image_relpath=image_relpath,
        cond_image_relpath=cond_image_relpath,
        image_path=metadata.resolve_asset_path(image_relpath),
        cond_image_path=metadata.resolve_asset_path(cond_image_relpath),
        task_ids=tuple(task_ids),
        task_strengths=normalized_strengths,
        source_sample_id=payload.get("source_sample_id"),
        source_task=payload.get("source_task"),
        weight=weight,
        metadata=resolved_metadata,
    )


def _build_derived_contract_records(
    source_records,
    *,
    dataset_root: Path,
    split: str,
    dataset_id: str,
    task_id: str,
    primitive_family: str,
) -> tuple[PreparedPilotRecord, ...]:
    prepared_records: list[PreparedPilotRecord] = []
    for source_record in source_records:
        prompt = read_prompt_text(source_record.prompt_path)
        cache_paths = build_preprocessed_sample_paths(
            source_record.dataset_root,
            dataset_name=source_record.dataset_name,
            sample_id=source_record.sample_id,
        )
        metadata_payload = {
            "runtime_view": PREPARED_RECORD_SOURCE_DERIVED_CONTRACT,
            "contract_dataset_id": dataset_id,
            "contract_split": split,
            "contract_sample_id": source_record.sample_id,
            "preprocessed": {
                "input_image_latent_relpath": _relative_asset_path(cache_paths.input_image_latent_path, dataset_root),
                "target_image_latent_relpath": _relative_asset_path(cache_paths.target_image_latent_path, dataset_root),
                "prompt_latent_relpath": _relative_asset_path(cache_paths.prompt_latent_path, dataset_root),
            },
        }
        input_relpath = _relative_asset_path(source_record.input_image_path, dataset_root)
        target_relpath = _relative_asset_path(source_record.target_image_path, dataset_root)

        prepared_records.append(
            PreparedPilotRecord(
                dataset_id=dataset_id,
                split=split,
                sample_id=f"{source_record.sample_id}__task",
                prompt=prompt,
                source_prompt=prompt,
                mode="task",
                primitive_family=primitive_family,
                image_relpath=target_relpath,
                cond_image_relpath=input_relpath,
                image_path=source_record.target_image_path,
                cond_image_path=source_record.input_image_path,
                task_ids=(task_id,),
                task_strengths=(1.0,),
                source_sample_id=source_record.sample_id,
                source_task=task_id,
                weight=1.0,
                metadata=metadata_payload,
            )
        )
        prepared_records.append(
            PreparedPilotRecord(
                dataset_id=dataset_id,
                split=split,
                sample_id=f"{source_record.sample_id}__identity",
                prompt=prompt,
                source_prompt=prompt,
                mode="identity",
                primitive_family=primitive_family,
                image_relpath=input_relpath,
                cond_image_relpath=input_relpath,
                image_path=source_record.input_image_path,
                cond_image_path=source_record.input_image_path,
                task_ids=(),
                task_strengths=(),
                source_sample_id=source_record.sample_id,
                source_task="identity",
                weight=1.0,
                metadata=metadata_payload,
            )
        )
    return tuple(prepared_records)


def _resolve_dataset_path(dataset_root: Path, relative_path: str) -> Path:
    candidate = (dataset_root / relative_path).resolve()
    try:
        candidate.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(f"Prepared dataset path '{relative_path}' escapes dataset root '{dataset_root}'") from error
    if not candidate.is_file():
        raise FileNotFoundError(f"Prepared dataset asset not found: {candidate}")
    return candidate


def _resolve_prepared_or_raw_path(dataset_root: Path, raw_root: Path | None, relative_path: str) -> Path:
    try:
        return _resolve_dataset_path(dataset_root, relative_path)
    except FileNotFoundError:
        if raw_root is None:
            raise
        return _resolve_dataset_path(raw_root, relative_path)


def _discover_raw_root(dataset_root: Path) -> Path | None:
    candidate = dataset_root.parent / "raw"
    return candidate if candidate.is_dir() else None


def _is_contract_split_root(split_root: Path) -> bool:
    required_dirs = (
        split_root / "raw" / "images" / "input",
        split_root / "raw" / "images" / "target",
        split_root / "raw" / "prompt",
        split_root / "preprocessed" / "image_latent_cache" / "input",
        split_root / "preprocessed" / "image_latent_cache" / "target",
        split_root / "preprocessed" / "prompt_latent_cache",
    )
    return all(path.is_dir() for path in required_dirs)


def _resolve_contract_runtime_spec(dataset_root: Path) -> dict[str, str]:
    dataset_name = dataset_root.name
    try:
        task_spec = get_task_spec_for_dataset_id(dataset_name)
    except KeyError as error:
        raise NotImplementedError(
            f"Automatic prepared/runtime derivation is not implemented for dataset root '{dataset_name}'."
        ) from error
    return {
        "task_id": task_spec.task_id,
        "primitive_family": task_spec.primitive_group,
    }


def _relative_asset_path(path: Path, dataset_root: Path) -> str:
    return str(path.resolve().relative_to(dataset_root))


def _resolve_preprocessed_cache_paths(
    record: PreparedPilotRecord,
    *,
    dataset_root: Path,
) -> dict[str, Path]:
    preprocessed_payload = record.metadata.get("preprocessed")
    context = f"{record.dataset_id}:{record.split}:{record.sample_id}"
    if not isinstance(preprocessed_payload, Mapping):
        raise ValueError(
            f"Prepared record '{context}' is missing 'metadata.preprocessed', which is required for "
            "backend='preprocessed'."
        )

    required_fields = {
        "input": "input_image_latent_relpath",
        "target": "target_image_latent_relpath",
        "prompt": "prompt_latent_relpath",
    }
    resolved: dict[str, Path] = {}
    for key, field_name in required_fields.items():
        relative_path = preprocessed_payload.get(field_name)
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(
                f"Prepared record '{context}' is missing 'metadata.preprocessed.{field_name}', which is required for "
                "backend='preprocessed'."
            )
        resolved[key] = _resolve_dataset_path(dataset_root, relative_path)
    return resolved


def _load_cached_latent_tensor(path: Path, *, context: str) -> Tensor:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Tensor):
        raise TypeError(f"Expected cached latent tensor at '{path}' for '{context}', received {type(payload).__name__}")
    if payload.ndim == 4 and payload.shape[0] == 1:
        payload = payload[0]
    if payload.ndim != 3:
        raise ValueError(f"Cached latent tensor at '{path}' for '{context}' must have shape [C, H, W]")
    return payload.detach().contiguous().float()


def _load_prompt_cache_payload(path: Path) -> dict[str, Tensor]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected prompt cache payload at '{path}' to be a mapping")

    prompt_embeds = payload.get("prompt_embeds")
    text_ids = payload.get("text_ids")
    if not isinstance(prompt_embeds, Tensor) or not isinstance(text_ids, Tensor):
        raise ValueError(f"Prompt cache payload at '{path}' must contain tensor fields 'prompt_embeds' and 'text_ids'")

    if prompt_embeds.ndim == 3 and prompt_embeds.shape[0] == 1:
        prompt_embeds = prompt_embeds[0]
    if text_ids.ndim == 3 and text_ids.shape[0] == 1:
        text_ids = text_ids[0]
    if prompt_embeds.ndim != 2:
        raise ValueError(f"Prompt cache field 'prompt_embeds' at '{path}' must have shape [seq_len, hidden_size]")
    if text_ids.ndim != 2:
        raise ValueError(f"Prompt cache field 'text_ids' at '{path}' must have shape [seq_len, 4]")
    if prompt_embeds.shape[0] != text_ids.shape[0]:
        raise ValueError(
            f"Prompt cache payload at '{path}' has mismatched sequence lengths: "
            f"{prompt_embeds.shape[0]} vs {text_ids.shape[0]}"
        )
    return {
        "prompt_embeds": prompt_embeds.detach().contiguous().float(),
        "text_ids": text_ids.detach().contiguous().long(),
    }


def _require_string(payload: Mapping[str, Any], key: str, *, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must provide a non-empty string for '{key}'")
    return value


def _normalize_size(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        if size <= 0:
            raise ValueError("size must be positive")
        return (size, size)

    if len(size) != 2:
        raise ValueError("size must be an int or a (height, width) tuple")
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError("size dimensions must be positive")
    return (height, width)


def _normalize_buckets(
    buckets: Sequence[tuple[int, int]] | None,
    *,
    default_size: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    if buckets is None:
        return (default_size,)

    normalized: list[tuple[int, int]] = []
    for bucket in buckets:
        if len(bucket) != 2:
            raise ValueError("Each bucket must be a (height, width) tuple")
        height, width = int(bucket[0]), int(bucket[1])
        if height <= 0 or width <= 0:
            raise ValueError("Bucket dimensions must be positive")
        normalized.append((height, width))
    if not normalized:
        raise ValueError("buckets cannot be empty")
    return tuple(normalized)


def _load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image = exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        else:
            image = image.copy()
    return image


def _paired_transform(
    image: Image.Image,
    cond_image: Image.Image,
    *,
    size: tuple[int, int],
    center_crop: bool,
    random_flip: bool,
) -> tuple[Tensor, Tensor]:
    target_width = size[1]
    target_height = size[0]
    image = image.resize((target_width, target_height), resample=Image.BILINEAR)
    cond_image = cond_image.resize((target_width, target_height), resample=Image.BILINEAR)

    if center_crop:
        image = _center_crop(image, size)
        cond_image = _center_crop(cond_image, size)

    if random_flip and random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        cond_image = cond_image.transpose(Image.FLIP_LEFT_RIGHT)

    return _image_to_tensor(image), _image_to_tensor(cond_image)


def _center_crop(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_height, target_width = size
    width, height = image.size
    left = max((width - target_width) // 2, 0)
    top = max((height - target_height) // 2, 0)
    return image.crop((left, top, left + target_width, top + target_height))


def _image_to_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.sub(0.5).div(0.5)


__all__ = [
    "COMPHOSER_DATA_BACKENDS",
    "PREPARED_PRIMITIVE_FORMAT_VERSION",
    "PREPARED_PRIMITIVE_METADATA_FILENAME",
    "PREPARED_RECORD_SOURCE_DERIVED_CONTRACT",
    "PREPARED_RECORD_SOURCE_MANIFEST",
    "PreparedPilotDataset",
    "PreparedPilotDatasetMetadata",
    "PreparedPilotRecord",
    "collate_prepared_pilot_examples",
    "find_nearest_bucket",
    "load_prepared_pilot_dataset_metadata",
    "load_prepared_pilot_records",
]

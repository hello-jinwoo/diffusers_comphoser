"""Checkpoint evaluation orchestration for ComPhoser primitive tasks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .controls import (
    PrimitiveTaskSpec,
    get_task_spec,
    get_task_spec_for_dataset_id,
    get_task_spec_for_primitive_group,
    normalize_primitive_groups,
)
from .datasets import load_prepared_pilot_records
from .inference import save_controlled_validation_artifacts
from .qformer import DEFAULT_QFORMER_NUM_LAYERS, DEFAULT_QFORMER_QUERY_COUNT, ComPhoserQFormer
from .training import (
    load_pilot_qformer_checkpoint,
    resolve_pilot_qformer_checkpoint_paths,
    validate_pilot_qformer_checkpoint_metadata,
)

DEFAULT_EVALUATION_MODE = "lora_qformer"
DEFAULT_EVALUATION_SCOPE = "primitive"
DEFAULT_PRETRAINED_MODEL_NAME_OR_PATH = "black-forest-labs/FLUX.2-klein-4B"
EVALUATION_MODES = ("flux_only", "lora_qformer")


@dataclass(frozen=True)
class EvaluationDatasetResolution:
    dataset_root: Path
    split: str
    dataset_id: str
    task: PrimitiveTaskSpec


@dataclass(frozen=True)
class CheckpointEvaluationConfig:
    checkpoint_dir: Path | None
    dataset_root: Path
    output_dir: Path
    evaluation_mode: str = DEFAULT_EVALUATION_MODE
    split: str = "val"
    sample_limit: int | None = None
    num_outputs_per_sample: int = 1
    num_inference_steps: int = 8
    seed: int | None = 17
    resolution: int | None = None
    guidance_scale: float = 3.5
    torch_dtype: str = "auto"
    device: str = "auto"
    pretrained_model_name_or_path: str | None = None
    revision: str | None = None
    variant: str | None = None
    primitive_group: str | None = None
    task_id: str | None = None
    qformer_num_heads: int = 16
    enable_model_cpu_offload: bool = True
    explicit_token_masking: tuple[float, ...] | None = None


def evaluate_checkpoint(config: CheckpointEvaluationConfig) -> dict[str, Any]:
    evaluation_mode = resolve_evaluation_mode(config.evaluation_mode)
    checkpoint_dir = None if config.checkpoint_dir is None else Path(config.checkpoint_dir).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = resolve_evaluation_dataset(
        config.dataset_root,
        split=config.split,
        primitive_group=config.primitive_group,
        task_id=config.task_id,
    )
    if evaluation_mode == "lora_qformer":
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required for lora_qformer evaluation")
        metadata = read_checkpoint_metadata(checkpoint_dir)
        _validate_checkpoint_supports_group(metadata, dataset.task.primitive_group)
        pretrained_model_name_or_path = str(config.pretrained_model_name_or_path or metadata["backbone_id"])
    else:
        if checkpoint_dir is not None:
            raise ValueError("checkpoint_dir is not used for flux_only evaluation; omit it")
        pretrained_model_name_or_path = str(config.pretrained_model_name_or_path or DEFAULT_PRETRAINED_MODEL_NAME_OR_PATH)
        metadata = {"backbone_id": pretrained_model_name_or_path}

    device = resolve_evaluation_device(config.device)
    torch_dtype = resolve_torch_dtype(config.torch_dtype, device=device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    pipeline = load_evaluation_pipeline(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        checkpoint_dir=checkpoint_dir,
        load_lora_weights=evaluation_mode == "lora_qformer",
        torch_dtype=torch_dtype,
        device=device,
        revision=config.revision,
        variant=config.variant,
        enable_model_cpu_offload=config.enable_model_cpu_offload,
    )
    qformer = None
    if evaluation_mode == "lora_qformer":
        qformer = load_evaluation_qformer(
            checkpoint_dir,
            metadata=metadata,
            device=device,
            torch_dtype=torch_dtype,
            num_heads=config.qformer_num_heads,
        )

    records = load_prepared_pilot_records(dataset.dataset_root, split=dataset.split)
    summary_path, controlled_summary = save_controlled_validation_artifacts(
        output_dir,
        pipelines_by_mode={evaluation_mode: pipeline},
        records=records,
        task_id=dataset.task.task_id,
        qformer=qformer,
        seed=config.seed,
        validation_mode=evaluation_mode,
        sample_limit=config.sample_limit,
        num_outputs_per_sample=config.num_outputs_per_sample,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        height=config.resolution,
        width=config.resolution,
        explicit_token_masking=config.explicit_token_masking,
        artifact_subdir="controlled_validation",
        nest_under_comphoser=False,
    )

    metrics_payload = build_metrics_report_payload(
        config=config,
        dataset=dataset,
        checkpoint_metadata=metadata,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        controlled_summary_path=summary_path,
        controlled_summary=controlled_summary,
    )
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    summary_md_path = output_dir / "summary.md"
    summary_md_path.write_text(build_summary_markdown(metrics_payload), encoding="utf-8")

    return {
        "output_dir": output_dir,
        "metrics_path": metrics_path,
        "summary_md_path": summary_md_path,
        "controlled_validation_summary_path": summary_path,
        "metrics": metrics_payload,
        "controlled_validation_summary": controlled_summary,
    }


def resolve_evaluation_dataset(
    dataset_root: str | Path,
    *,
    split: str = "val",
    primitive_group: str | None = None,
    task_id: str | None = None,
) -> EvaluationDatasetResolution:
    requested_path = Path(dataset_root).expanduser().resolve()
    requested_split = str(split)
    if requested_split not in {"train", "val"}:
        raise ValueError("Evaluation split must be 'train' or 'val'")

    if requested_path.name in {"train", "val"}:
        if requested_path.name != requested_split:
            raise ValueError(
                f"Dataset path points at split '{requested_path.name}' but --split requested '{requested_split}'"
            )
        resolved_split = requested_path.name
        resolved_dataset_root = requested_path.parent
    else:
        resolved_split = requested_split
        resolved_dataset_root = requested_path

    if task_id is not None:
        task = get_task_spec(task_id)
    elif primitive_group is not None:
        task = get_task_spec_for_primitive_group(primitive_group)
        if task is None:
            raise ValueError(f"No dataset-backed task is registered for primitive group '{primitive_group}'")
    else:
        task = get_task_spec_for_dataset_id(resolved_dataset_root.name)

    if primitive_group is not None:
        expected_group = normalize_primitive_groups((primitive_group,))[0]
        if task.primitive_group != expected_group:
            raise ValueError(
                f"Task '{task.task_id}' belongs to primitive group '{task.primitive_group}', "
                f"not requested group '{expected_group}'"
            )

    split_root = resolved_dataset_root / resolved_split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Evaluation dataset split root not found: {split_root}")

    return EvaluationDatasetResolution(
        dataset_root=resolved_dataset_root,
        split=resolved_split,
        dataset_id=resolved_dataset_root.name,
        task=task,
    )


def read_checkpoint_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
    paths = resolve_pilot_qformer_checkpoint_paths(checkpoint_dir)
    if not paths.metadata_path.is_file():
        raise FileNotFoundError(f"Missing ComPhoser checkpoint metadata: {paths.metadata_path}")
    with paths.metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    validate_pilot_qformer_checkpoint_metadata(metadata)
    return metadata


def load_evaluation_qformer(
    checkpoint_dir: str | Path,
    *,
    metadata: Mapping[str, Any],
    device: torch.device,
    torch_dtype: torch.dtype,
    num_heads: int = 16,
) -> ComPhoserQFormer:
    qformer = ComPhoserQFormer(
        hidden_size=int(metadata["query_hidden_size"]),
        cond_token_dim=int(metadata.get("cond_token_dim", metadata["query_hidden_size"])),
        num_queries=int(metadata.get("query_count", DEFAULT_QFORMER_QUERY_COUNT)),
        num_layers=int(metadata.get("num_layers", DEFAULT_QFORMER_NUM_LAYERS)),
        num_heads=int(num_heads),
    )
    load_pilot_qformer_checkpoint(checkpoint_dir, qformer=qformer)
    qformer.requires_grad_(False)
    qformer.eval()
    return qformer.to(device=device, dtype=torch_dtype)


def load_evaluation_pipeline(
    *,
    pretrained_model_name_or_path: str,
    checkpoint_dir: str | Path | None,
    load_lora_weights: bool = True,
    torch_dtype: torch.dtype,
    device: torch.device,
    revision: str | None = None,
    variant: str | None = None,
    enable_model_cpu_offload: bool = True,
):
    from diffusers import Flux2KleinPipeline

    pipeline = Flux2KleinPipeline.from_pretrained(
        pretrained_model_name_or_path,
        revision=revision,
        variant=variant,
        torch_dtype=torch_dtype,
    )
    if load_lora_weights:
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required when load_lora_weights=True")
        pipeline.load_lora_weights(str(checkpoint_dir))
    if enable_model_cpu_offload and device.type == "cuda":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def resolve_evaluation_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_evaluation_mode(evaluation_mode: str) -> str:
    if evaluation_mode in EVALUATION_MODES:
        return evaluation_mode
    supported = ", ".join(EVALUATION_MODES)
    raise ValueError(f"Unsupported evaluation_mode '{evaluation_mode}'. Expected one of: {supported}")


def resolve_torch_dtype(dtype_name: str, *, device: torch.device) -> torch.dtype:
    if dtype_name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return dtype_map[dtype_name]
    except KeyError as error:
        supported = ", ".join(("auto", *dtype_map))
        raise ValueError(f"Unsupported torch dtype '{dtype_name}'. Expected one of: {supported}") from error


def build_metrics_report_payload(
    *,
    config: CheckpointEvaluationConfig,
    dataset: EvaluationDatasetResolution,
    checkpoint_metadata: Mapping[str, Any],
    pretrained_model_name_or_path: str,
    controlled_summary_path: Path,
    controlled_summary: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = Path(config.output_dir).expanduser().resolve()
    checkpoint_dir = None if config.checkpoint_dir is None else str(Path(config.checkpoint_dir).expanduser().resolve())
    return {
        "artifact_version": "comphoser-primitive-checkpoint-evaluation-v1",
        "evaluation_mode": resolve_evaluation_mode(config.evaluation_mode),
        "checkpoint_dir": checkpoint_dir,
        "dataset_root": str(dataset.dataset_root),
        "dataset_id": dataset.dataset_id,
        "split": dataset.split,
        "task_id": dataset.task.task_id,
        "primitive_group": dataset.task.primitive_group,
        "backbone_id": checkpoint_metadata.get("backbone_id", pretrained_model_name_or_path),
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "runtime": {
            "seed": config.seed,
            "sample_limit": config.sample_limit,
            "num_outputs_per_sample": config.num_outputs_per_sample,
            "num_inference_steps": config.num_inference_steps,
            "resolution": config.resolution,
            "guidance_scale": config.guidance_scale,
            "torch_dtype": config.torch_dtype,
            "device": config.device,
            "enable_model_cpu_offload": config.enable_model_cpu_offload,
            "explicit_token_masking": None
            if config.explicit_token_masking is None
            else list(config.explicit_token_masking),
        },
        "controlled_validation_summary": str(controlled_summary_path.relative_to(output_dir)),
        "controlled_validation_artifact_version": controlled_summary.get("artifact_version"),
        "case_count": controlled_summary.get("case_count"),
        "sample_count": controlled_summary.get("sample_count"),
        "run_count": controlled_summary.get("run_count"),
        "metrics": controlled_summary.get("metrics", {}),
    }


def build_summary_markdown(metrics_payload: Mapping[str, Any]) -> str:
    lines = [
        "# Primitive Evaluation",
        "",
        f"- mode: `{metrics_payload['evaluation_mode']}`",
        f"- checkpoint: `{metrics_payload['checkpoint_dir']}`",
        f"- pretrained_model: `{metrics_payload.get('pretrained_model_name_or_path', metrics_payload.get('backbone_id'))}`",
        f"- dataset: `{metrics_payload['dataset_root']}`",
        f"- split: `{metrics_payload['split']}`",
        f"- task: `{metrics_payload['task_id']}`",
        f"- primitive_group: `{metrics_payload['primitive_group']}`",
        f"- samples: `{metrics_payload['sample_count']}`",
        f"- outputs: `{metrics_payload['run_count']}`",
        "",
        "## Metrics",
        "",
    ]
    for metric_name, summary in sorted(metrics_payload.get("metrics", {}).items()):
        if not isinstance(summary, Mapping) or summary.get("status") != "available":
            lines.append(f"- {metric_name}: unavailable")
            continue
        mean = float(summary["mean"])
        unit = summary.get("unit", "")
        lines.append(f"- {metric_name}: mean={mean:.6g} {unit}, count={summary.get('count')}")
    lines.append("")
    lines.append(f"Controlled validation summary: `{metrics_payload['controlled_validation_summary']}`")
    lines.append("")
    return "\n".join(lines)


def _validate_checkpoint_supports_group(metadata: Mapping[str, Any], primitive_group: str) -> None:
    checkpoint_groups = normalize_primitive_groups(metadata.get("primitive_groups", ()))
    target_group = normalize_primitive_groups((primitive_group,))[0]
    if target_group not in checkpoint_groups:
        raise ValueError(
            f"Checkpoint primitive_groups {checkpoint_groups} do not include target primitive group '{target_group}'"
        )


def config_to_jsonable(config: CheckpointEvaluationConfig) -> dict[str, Any]:
    payload = asdict(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


__all__ = [
    "CheckpointEvaluationConfig",
    "DEFAULT_EVALUATION_MODE",
    "DEFAULT_EVALUATION_SCOPE",
    "DEFAULT_PRETRAINED_MODEL_NAME_OR_PATH",
    "EVALUATION_MODES",
    "EvaluationDatasetResolution",
    "build_metrics_report_payload",
    "build_summary_markdown",
    "config_to_jsonable",
    "evaluate_checkpoint",
    "load_evaluation_pipeline",
    "load_evaluation_qformer",
    "read_checkpoint_metadata",
    "resolve_evaluation_dataset",
    "resolve_evaluation_device",
    "resolve_evaluation_mode",
    "resolve_torch_dtype",
]

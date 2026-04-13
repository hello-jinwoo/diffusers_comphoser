"""Inference-facing helpers for ComPhoser."""

from __future__ import annotations

from contextlib import ExitStack
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch import Tensor

from .controls import ResolvedPrimitiveSelection, resolve_control_selection
from .datasets import PreparedPilotRecord
from .qformer import ComPhoserQFormer
from .training import prepare_pilot_transformer_conditioning

CONTROLLED_VALIDATION_COMPARISON_MODES = ("flux_only", "lora_only", "lora_qformer")
CONTROLLED_VALIDATION_ARTIFACT_SUBDIR = "controlled_validation"
CONTROLLED_VALIDATION_ARTIFACT_VERSION = "comphoser-controlled-validation-v3"
DEFAULT_CONTROLLED_VALIDATION_STEPS = 8
DEFAULT_CONTROLLED_VALIDATION_SEEDS_PER_IMAGE = 2
DEFAULT_CONTROLLED_VALIDATION_PROMPT_VARIANTS = (
    "restore fine detail and improve apparent sharpness while preserving natural color and composition",
    "restore fine detail conservatively while preserving natural color and composition",
    "restore fine detail more strongly while suppressing artifacts and preserving natural color and composition",
)


@dataclass(frozen=True)
class InferenceControlSpec:
    mode: str
    controls: ResolvedPrimitiveSelection

    @property
    def uses_qformer(self) -> bool:
        return self.mode == "controlled"


@dataclass(frozen=True)
class ControlledValidationCase:
    case_id: str
    case_type: str
    sample_id: str
    source_sample_id: str | None
    prompt: str
    source_prompt: str | None
    primitive_family: str | None
    condition_image_path: Path
    target_image_path: Path | None
    source_task: str | None
    task_strength: float


@dataclass(frozen=True)
class PilotInferenceConditioning:
    prompt_embeds: Tensor
    negative_prompt_embeds: Tensor | None
    added_token_count: int
    height: int
    width: int
    raw_query_gates: Tensor | None = None
    query_gates: Tensor | None = None


def resolve_inference_control(
    primitive_groups: Sequence[str] | str | None = None,
    task_strengths: Mapping[str, float] | Sequence[float] | float | None = None,
) -> InferenceControlSpec:
    controls = resolve_control_selection(primitive_groups=primitive_groups, task_strengths=task_strengths)
    mode = "controlled" if controls.is_control_enabled else "baseline"
    return InferenceControlSpec(mode=mode, controls=controls)


def build_controlled_validation_prompt_panel(source_prompt: str | None = None) -> tuple[str, ...]:
    base_prompt = source_prompt.strip() if source_prompt else DEFAULT_CONTROLLED_VALIDATION_PROMPT_VARIANTS[0]
    prompts: list[str] = []
    for candidate in (base_prompt, *DEFAULT_CONTROLLED_VALIDATION_PROMPT_VARIANTS[1:]):
        normalized = candidate.strip()
        if normalized and normalized not in prompts:
            prompts.append(normalized)
    return tuple(prompts)


def build_controlled_validation_cases(
    records: Sequence[PreparedPilotRecord],
    *,
    sample_limit: int | None = None,
) -> tuple[ControlledValidationCase, ...]:
    task_records = tuple(record for record in records if record.mode == "task")
    if not task_records:
        raise ValueError("Controlled validation requires at least one prepared task record")
    if sample_limit is not None and sample_limit <= 0:
        raise ValueError("Controlled validation sample_limit must be positive when provided")

    cases: list[ControlledValidationCase] = []
    selected_task_records = sorted(task_records, key=_controlled_validation_record_sort_key)
    if sample_limit is not None:
        selected_task_records = selected_task_records[:sample_limit]
    for task_record in selected_task_records:
        cases.append(
            _build_controlled_validation_case(
                case_id=_build_controlled_validation_case_prefix(task_record),
                case_type="sample",
                record=task_record,
                prompt=task_record.prompt,
                task_strength=_resolve_record_task_strength(task_record),
            )
        )

    return tuple(cases)


def build_single_validation_case(
    *,
    prompt: str,
    condition_image_path: str | Path,
    target_image_path: str | Path | None = None,
    sample_id: str = "single_case",
    source_sample_id: str | None = None,
    source_prompt: str | None = None,
    primitive_family: str | None = None,
    source_task: str | None = None,
    task_strength: float = 1.0,
) -> ControlledValidationCase:
    condition_path = Path(condition_image_path).expanduser().resolve()
    target_path = None if target_image_path is None else Path(target_image_path).expanduser().resolve()
    return ControlledValidationCase(
        case_id=_normalize_case_id(sample_id),
        case_type="single",
        sample_id=sample_id,
        source_sample_id=source_sample_id,
        prompt=prompt,
        source_prompt=source_prompt,
        primitive_family=primitive_family,
        condition_image_path=condition_path,
        target_image_path=target_path,
        source_task=source_task,
        task_strength=float(task_strength),
    )


def prepare_pilot_inference_conditioning(
    pipeline,
    prompt: str,
    condition_image: Image.Image,
    *,
    qformer: ComPhoserQFormer | None,
    task_id: str | None = None,
    task_strength: float = 1.0,
    negative_prompt: str = "",
    generator: torch.Generator | None = None,
    guidance_scale: float = 3.5,
    height: int | None = None,
    width: int | None = None,
    max_sequence_length: int = 512,
    text_encoder_out_layers: Sequence[int] = (9, 18, 27),
) -> PilotInferenceConditioning:
    device = _resolve_pipeline_device(pipeline)
    prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        text_encoder_out_layers=tuple(text_encoder_out_layers),
    )

    negative_prompt_embeds = None
    if guidance_scale > 1.0 and not getattr(pipeline.config, "is_distilled", False):
        negative_prompt_embeds, _ = pipeline.encode_prompt(
            prompt=negative_prompt,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=tuple(text_encoder_out_layers),
        )

    prepared_condition_image, resolved_height, resolved_width = _prepare_condition_image_tensor(
        pipeline,
        condition_image,
        height=height,
        width=width,
    )

    if qformer is None:
        return PilotInferenceConditioning(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            added_token_count=0,
            height=resolved_height,
            width=resolved_width,
        )

    if task_id is None:
        raise ValueError("task_id is required when Q-Former inference conditioning is enabled")

    cond_tokens, _ = pipeline.prepare_image_latents(
        images=[prepared_condition_image],
        batch_size=1,
        generator=generator,
        device=device,
        dtype=pipeline.vae.dtype,
    )
    conditioning = prepare_pilot_transformer_conditioning(
        prompt_embeds,
        text_ids,
        cond_tokens,
        qformer=qformer,
        task_id=task_id,
        task_strengths=(float(task_strength),),
    )
    return PilotInferenceConditioning(
        prompt_embeds=conditioning.encoder_hidden_states,
        negative_prompt_embeds=negative_prompt_embeds,
        added_token_count=conditioning.added_token_count,
        height=resolved_height,
        width=resolved_width,
        raw_query_gates=conditioning.raw_query_gates,
        query_gates=conditioning.query_gates,
    )


def run_controlled_validation_case(
    pipeline,
    case: ControlledValidationCase,
    *,
    mode: str,
    qformer: ComPhoserQFormer | None,
    task_id: str | None,
    seed: int | None,
    num_inference_steps: int = DEFAULT_CONTROLLED_VALIDATION_STEPS,
    guidance_scale: float = 3.5,
    height: int | None = None,
    width: int | None = None,
    max_sequence_length: int = 512,
    text_encoder_out_layers: Sequence[int] = (9, 18, 27),
) -> tuple[Image.Image, dict[str, Any]]:
    if mode not in CONTROLLED_VALIDATION_COMPARISON_MODES:
        supported = ", ".join(CONTROLLED_VALIDATION_COMPARISON_MODES)
        raise ValueError(f"Unsupported controlled validation mode '{mode}'. Expected one of: {supported}")
    controller_enabled = mode == "lora_qformer"
    if controller_enabled and qformer is None:
        raise ValueError("lora_qformer controlled validation requires a Q-Former instance")

    condition_image = _load_rgb_image(case.condition_image_path)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=_resolve_pipeline_device(pipeline)).manual_seed(seed)

    with _controlled_validation_inference_context(pipeline):
        conditioning = prepare_pilot_inference_conditioning(
            pipeline,
            case.prompt,
            condition_image,
            qformer=qformer if controller_enabled else None,
            task_id=task_id if controller_enabled else None,
            task_strength=case.task_strength,
            generator=generator,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )
        image = pipeline(
            image=condition_image,
            prompt_embeds=conditioning.prompt_embeds,
            negative_prompt_embeds=conditioning.negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=conditioning.height,
            width=conditioning.width,
        ).images[0]

    summary = {
        "mode": mode,
        "controller_enabled": bool(controller_enabled),
        "controller_path": _resolve_controller_path(mode, controller_enabled),
        "case_id": case.case_id,
        "case_type": case.case_type,
        "sample_id": case.sample_id,
        "source_sample_id": case.source_sample_id,
        "prompt": case.prompt,
        "source_prompt": case.source_prompt,
        "primitive_family": case.primitive_family,
        "condition_image_path": str(case.condition_image_path),
        "target_image_path": None if case.target_image_path is None else str(case.target_image_path),
        "source_task": case.source_task,
        "task_strength": case.task_strength,
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "height": conditioning.height,
        "width": conditioning.width,
        "added_token_count": conditioning.added_token_count,
    }
    summary.update(_summarize_query_gates(conditioning.raw_query_gates, conditioning.query_gates))
    return image, summary


def save_validation_artifacts(
    output_dir: str | Path,
    *,
    pipelines_by_mode: Mapping[str, Any],
    cases: Sequence[ControlledValidationCase],
    task_id: str,
    qformer: ComPhoserQFormer | None,
    seed: int | None,
    validation_mode: str | None = None,
    num_outputs_per_sample: int = DEFAULT_CONTROLLED_VALIDATION_SEEDS_PER_IMAGE,
    num_inference_steps: int = DEFAULT_CONTROLLED_VALIDATION_STEPS,
    guidance_scale: float = 3.5,
    height: int | None = None,
    width: int | None = None,
    max_sequence_length: int = 512,
    text_encoder_out_layers: Sequence[int] = (9, 18, 27),
    artifact_subdir: str = CONTROLLED_VALIDATION_ARTIFACT_SUBDIR,
) -> tuple[Path, dict[str, Any]]:
    if not cases:
        raise ValueError("Validation artifacts require at least one case")
    if num_outputs_per_sample <= 0:
        raise ValueError("Validation artifacts require num_outputs_per_sample to be positive")

    active_mode = _resolve_active_validation_mode(
        pipelines_by_mode=pipelines_by_mode,
        validation_mode=validation_mode,
    )

    artifact_dir = Path(output_dir).expanduser().resolve() / "comphoser" / artifact_subdir
    (artifact_dir / "images" / active_mode).mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        condition_image = _load_rgb_image(case.condition_image_path)
        target_image = _load_reference_or_blank_image(case.target_image_path, fallback_size=condition_image.size)

        input_relpath = _build_validation_reference_image_path(mode=active_mode, case_id=case.case_id, kind="input")
        gt_relpath = _build_validation_reference_image_path(mode=active_mode, case_id=case.case_id, kind="gt")
        _save_rgb_image(artifact_dir / input_relpath, condition_image)
        _save_rgb_image(artifact_dir / gt_relpath, target_image)

        output_summaries: list[dict[str, Any]] = []
        output_images: list[Image.Image] = []
        output_relpaths: list[str] = []
        for output_index in range(num_outputs_per_sample):
            output_seed = _resolve_validation_output_seed(
                seed=seed,
                case_index=case_index,
                output_index=output_index,
                num_outputs_per_sample=num_outputs_per_sample,
            )
            image, run_summary = run_controlled_validation_case(
                pipelines_by_mode[active_mode],
                case,
                mode=active_mode,
                qformer=qformer if active_mode == "lora_qformer" else None,
                task_id=task_id,
                seed=output_seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )
            relative_image_path = _build_validation_output_image_path(
                mode=active_mode,
                case_id=case.case_id,
                output_index=output_index + 1,
            )
            _save_rgb_image(artifact_dir / relative_image_path, image)
            serialized_run = _serialize_validation_run(
                run_summary,
                image_path=str(relative_image_path),
                output_index=output_index + 1,
            )
            output_summaries.append(serialized_run)
            output_images.append(image.copy())
            output_relpaths.append(str(relative_image_path))
            runs.append(serialized_run)

        all_relpath = _build_validation_reference_image_path(mode=active_mode, case_id=case.case_id, kind="all")
        contact_sheet = _build_validation_contact_sheet(condition_image, output_images, target_image)
        _save_rgb_image(artifact_dir / all_relpath, contact_sheet)

        samples.append(
            {
                "image_id": case.case_id,
                "sample_id": case.sample_id,
                "source_sample_id": case.source_sample_id,
                "prompt": case.prompt,
                "source_prompt": case.source_prompt,
                "primitive_family": case.primitive_family,
                "source_task": case.source_task,
                "task_strength": case.task_strength,
                "source_paths": {
                    "condition_image_path": str(case.condition_image_path),
                    "target_image_path": None if case.target_image_path is None else str(case.target_image_path),
                },
                "ground_truth_present": case.target_image_path is not None,
                "output_image_count": len(output_summaries),
                "artifacts": {
                    "input_image": str(input_relpath),
                    "output_images": output_relpaths,
                    "ground_truth_image": str(gt_relpath),
                    "contact_sheet_image": str(all_relpath),
                },
                "outputs": output_summaries,
            }
        )

    summary_payload = {
        "artifact_version": CONTROLLED_VALIDATION_ARTIFACT_VERSION,
        "task_id": task_id,
        "active_validation_mode": active_mode,
        "case_count": len(cases),
        "sample_count": len(cases),
        "run_count": len(runs),
        "num_validation_seeds_per_image": num_outputs_per_sample,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "cases": [_serialize_controlled_validation_case(case) for case in cases],
        "samples": samples,
    }
    summary_path = artifact_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary_path, summary_payload


def save_controlled_validation_artifacts(
    output_dir: str | Path,
    *,
    pipelines_by_mode: Mapping[str, Any],
    records: Sequence[PreparedPilotRecord],
    task_id: str,
    qformer: ComPhoserQFormer | None,
    seed: int | None,
    validation_mode: str | None = None,
    sample_limit: int | None = None,
    num_outputs_per_sample: int = DEFAULT_CONTROLLED_VALIDATION_SEEDS_PER_IMAGE,
    num_inference_steps: int = DEFAULT_CONTROLLED_VALIDATION_STEPS,
    guidance_scale: float = 3.5,
    height: int | None = None,
    width: int | None = None,
    max_sequence_length: int = 512,
    text_encoder_out_layers: Sequence[int] = (9, 18, 27),
    artifact_subdir: str = CONTROLLED_VALIDATION_ARTIFACT_SUBDIR,
) -> tuple[Path, dict[str, Any]]:
    return save_validation_artifacts(
        output_dir,
        pipelines_by_mode=pipelines_by_mode,
        cases=build_controlled_validation_cases(records, sample_limit=sample_limit),
        task_id=task_id,
        qformer=qformer,
        seed=seed,
        validation_mode=validation_mode,
        num_outputs_per_sample=num_outputs_per_sample,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        max_sequence_length=max_sequence_length,
        text_encoder_out_layers=text_encoder_out_layers,
        artifact_subdir=artifact_subdir,
    )


def _build_controlled_validation_case(
    *,
    case_id: str,
    case_type: str,
    record: PreparedPilotRecord,
    prompt: str,
    task_strength: float,
) -> ControlledValidationCase:
    return ControlledValidationCase(
        case_id=case_id,
        case_type=case_type,
        sample_id=record.sample_id,
        source_sample_id=record.source_sample_id,
        prompt=prompt,
        source_prompt=record.source_prompt,
        primitive_family=record.primitive_family,
        condition_image_path=record.cond_image_path,
        target_image_path=record.image_path,
        source_task=record.source_task,
        task_strength=float(task_strength),
    )


def _resolve_record_task_strength(record: PreparedPilotRecord) -> float:
    if not record.task_ids:
        return 0.0
    if len(record.task_ids) != 1 or len(record.task_strengths) != 1:
        raise NotImplementedError("Controlled validation only supports single-task prepared records")
    return float(record.task_strengths[0])


def _resolve_pipeline_device(pipeline) -> torch.device:
    device = getattr(pipeline, "_execution_device", None)
    if device is None:
        device = getattr(pipeline, "device", None)
    if device is None:
        raise ValueError("Could not resolve an execution device from the provided pipeline")
    return torch.device(device)


def _resolve_module_dtype(module: Any) -> torch.dtype | None:
    if module is None:
        return None

    dtype = getattr(module, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype

    for attr_name in ("parameters", "buffers"):
        accessor = getattr(module, attr_name, None)
        if not callable(accessor):
            continue
        try:
            first_tensor = next(accessor(), None)
        except TypeError:
            first_tensor = None
        if isinstance(first_tensor, Tensor):
            return first_tensor.dtype
    return None


def _resolve_pipeline_inference_dtype(pipeline) -> torch.dtype | None:
    for module_name in ("vae", "transformer", "text_encoder"):
        dtype = _resolve_module_dtype(getattr(pipeline, module_name, None))
        if dtype is not None:
            return dtype
    return None


def _controlled_validation_inference_context(pipeline):
    stack = ExitStack()
    stack.enter_context(torch.inference_mode())

    device_type = _resolve_pipeline_device(pipeline).type
    if device_type not in {"cuda", "cpu"}:
        return stack

    inference_dtype = _resolve_pipeline_inference_dtype(pipeline)
    if inference_dtype not in {torch.float16, torch.bfloat16}:
        return stack

    stack.enter_context(torch.autocast(device_type=device_type, dtype=inference_dtype))
    return stack


def _prepare_condition_image_tensor(
    pipeline,
    image: Image.Image,
    *,
    height: int | None,
    width: int | None,
) -> tuple[Tensor, int, int]:
    pipeline.image_processor.check_image_input(image)

    working_image = image
    image_width, image_height = working_image.size
    if image_width * image_height > 1024 * 1024:
        working_image = pipeline.image_processor._resize_to_target_area(working_image, 1024 * 1024)
        image_width, image_height = working_image.size

    multiple_of = pipeline.vae_scale_factor * 2
    resolved_width = int(width or image_width)
    resolved_height = int(height or image_height)
    resolved_width = max(multiple_of, (resolved_width // multiple_of) * multiple_of)
    resolved_height = max(multiple_of, (resolved_height // multiple_of) * multiple_of)

    prepared = pipeline.image_processor.preprocess(
        working_image,
        height=resolved_height,
        width=resolved_width,
        resize_mode="crop",
    )
    return prepared, resolved_height, resolved_width


def _summarize_query_gates(
    raw_query_gates: Tensor | None,
    query_gates: Tensor | None,
) -> dict[str, Any]:
    if raw_query_gates is None or query_gates is None:
        return {
            "raw_query_gates": None,
            "effective_query_gates": None,
            "active_query_gates": None,
            "raw_query_gate_mean": None,
            "raw_query_gate_std": None,
            "effective_query_gate_mean": None,
            "effective_query_gate_min": None,
            "effective_query_gate_max": None,
            "active_query_gate_mean": None,
            "active_query_gate_min": None,
            "active_query_gate_max": None,
        }

    raw_values = raw_query_gates.detach().cpu().reshape(raw_query_gates.shape[0], -1)[0]
    effective_values = query_gates.detach().cpu().reshape(query_gates.shape[0], -1)[0]
    return {
        "raw_query_gates": [float(value) for value in raw_values.tolist()],
        "effective_query_gates": [float(value) for value in effective_values.tolist()],
        "active_query_gates": [float(value) for value in effective_values.tolist()],
        "raw_query_gate_mean": float(raw_values.mean().item()),
        "raw_query_gate_std": float(raw_values.std(unbiased=False).item()),
        "effective_query_gate_mean": float(effective_values.mean().item()),
        "effective_query_gate_min": float(effective_values.min().item()),
        "effective_query_gate_max": float(effective_values.max().item()),
        "active_query_gate_mean": float(effective_values.mean().item()),
        "active_query_gate_min": float(effective_values.min().item()),
        "active_query_gate_max": float(effective_values.max().item()),
    }


def _serialize_controlled_validation_case(case: ControlledValidationCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "case_type": case.case_type,
        "sample_id": case.sample_id,
        "source_sample_id": case.source_sample_id,
        "prompt": case.prompt,
        "source_prompt": case.source_prompt,
        "primitive_family": case.primitive_family,
        "condition_image_path": str(case.condition_image_path),
        "target_image_path": None if case.target_image_path is None else str(case.target_image_path),
        "source_task": case.source_task,
        "task_strength": case.task_strength,
    }


def _load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image = exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        else:
            image = image.copy()
    return image


def _save_rgb_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(path)


def _build_validation_reference_image_path(*, mode: str, case_id: str, kind: str) -> Path:
    return Path("images") / mode / f"{case_id}_{kind}.png"


def _build_validation_output_image_path(*, mode: str, case_id: str, output_index: int) -> Path:
    return Path("images") / mode / f"{case_id}_output_{output_index}.png"


def _resolve_controller_path(mode: str, controller_enabled: bool) -> str:
    if mode != "lora_qformer":
        return "not_available"
    return "enabled" if controller_enabled else "disabled"


def _resolve_active_validation_mode(
    *,
    pipelines_by_mode: Mapping[str, Any],
    validation_mode: str | None,
) -> str:
    if validation_mode is not None:
        if validation_mode not in pipelines_by_mode:
            raise ValueError(f"Missing controlled validation pipeline for mode '{validation_mode}'")
        return validation_mode

    if len(pipelines_by_mode) != 1:
        available = ", ".join(sorted(pipelines_by_mode))
        raise ValueError(
            "Validation mode must be explicit when more than one pipeline is provided. "
            f"Available modes: {available}"
        )
    return next(iter(pipelines_by_mode))


def _resolve_validation_output_seed(
    *,
    seed: int | None,
    case_index: int,
    output_index: int,
    num_outputs_per_sample: int,
) -> int | None:
    if seed is None:
        return None
    return int(seed) + case_index * num_outputs_per_sample + output_index


def _serialize_validation_run(
    run_summary: Mapping[str, Any],
    *,
    image_path: str,
    output_index: int,
) -> dict[str, Any]:
    return {
        "output_index": output_index,
        "seed": run_summary["seed"],
        "image_path": image_path,
        "controller_enabled": run_summary["controller_enabled"],
        "controller_path": run_summary["controller_path"],
        "added_token_count": run_summary["added_token_count"],
        "height": run_summary["height"],
        "width": run_summary["width"],
        "raw_query_gates": run_summary["raw_query_gates"],
        "effective_query_gates": run_summary["effective_query_gates"],
        "active_query_gates": run_summary["active_query_gates"],
        "raw_query_gate_mean": run_summary["raw_query_gate_mean"],
        "raw_query_gate_std": run_summary["raw_query_gate_std"],
        "effective_query_gate_mean": run_summary["effective_query_gate_mean"],
        "effective_query_gate_min": run_summary["effective_query_gate_min"],
        "effective_query_gate_max": run_summary["effective_query_gate_max"],
        "active_query_gate_mean": run_summary["active_query_gate_mean"],
        "active_query_gate_min": run_summary["active_query_gate_min"],
        "active_query_gate_max": run_summary["active_query_gate_max"],
    }


def _load_reference_or_blank_image(
    path: Path | None,
    *,
    fallback_size: tuple[int, int],
) -> Image.Image:
    if path is None:
        return Image.new("RGB", fallback_size, color=(255, 255, 255))
    return _load_rgb_image(path)


def _build_validation_contact_sheet(
    condition_image: Image.Image,
    output_images: Sequence[Image.Image],
    target_image: Image.Image,
) -> Image.Image:
    if not output_images:
        raise ValueError("Validation contact sheet requires at least one generated output image")

    tile_size = output_images[0].size
    panel_images = [condition_image, *output_images, target_image]
    resized_images = [
        image if image.size == tile_size else image.resize(tile_size, resample=Image.BILINEAR)
        for image in panel_images
    ]
    sheet = Image.new("RGB", (tile_size[0] * len(resized_images), tile_size[1]))
    for index, image in enumerate(resized_images):
        sheet.paste(image, (index * tile_size[0], 0))
    return sheet


def _build_controlled_validation_case_prefix(record: PreparedPilotRecord) -> str:
    source_key = record.source_sample_id or record.sample_id
    return _normalize_case_id(source_key)


def _controlled_validation_record_sort_key(record: PreparedPilotRecord) -> tuple[str, str]:
    return (str(record.source_sample_id or ""), str(record.sample_id))


def _normalize_case_id(value: str) -> str:
    normalized = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in value)
    return normalized or "case"


__all__ = [
    "CONTROLLED_VALIDATION_COMPARISON_MODES",
    "CONTROLLED_VALIDATION_ARTIFACT_SUBDIR",
    "CONTROLLED_VALIDATION_ARTIFACT_VERSION",
    "DEFAULT_CONTROLLED_VALIDATION_SEEDS_PER_IMAGE",
    "DEFAULT_CONTROLLED_VALIDATION_PROMPT_VARIANTS",
    "DEFAULT_CONTROLLED_VALIDATION_STEPS",
    "ControlledValidationCase",
    "InferenceControlSpec",
    "PilotInferenceConditioning",
    "build_controlled_validation_cases",
    "build_controlled_validation_prompt_panel",
    "build_single_validation_case",
    "prepare_pilot_inference_conditioning",
    "resolve_inference_control",
    "run_controlled_validation_case",
    "save_validation_artifacts",
    "save_controlled_validation_artifacts",
]

"""ComPhoser-owned runtime helpers layered on top of the retained trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from typing import Any, Callable, Mapping, Sequence

import torch
from PIL import Image, ImageDraw
from peft import set_peft_model_state_dict

from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from diffusers.training_utils import free_memory

from .datasets import MultiPreparedPilotDataset, PreparedPilotDataset, load_prepared_pilot_records
from .inference import (
    DEFAULT_CONTROLLED_VALIDATION_STEPS,
    build_single_validation_case,
    save_validation_artifacts,
    save_controlled_validation_artifacts,
)
from .qformer import ComPhoserQFormer
from .training import (
    build_pilot_prompt_policy_summary,
    build_pilot_qformer_checkpoint_metadata,
    resolve_pilot_training_runtime,
    resolve_validation_inference_mode,
    save_pilot_qformer_checkpoint,
    update_controlled_validation_metadata,
)


def _detach_state_dict_to_cpu(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    detached: dict[str, Any] = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            detached[key] = value.detach().cpu().contiguous()
        else:
            detached[key] = value
    return detached


def build_detached_validation_qformer(
    qformer: ComPhoserQFormer,
    *,
    state_dict: Mapping[str, Any] | None = None,
) -> ComPhoserQFormer:
    validation_qformer = ComPhoserQFormer(
        hidden_size=qformer.hidden_size,
        cond_token_dim=qformer.cond_token_dim,
        num_queries=qformer.num_queries,
        cond_summary_tokens=qformer.cond_summary_tokens,
        num_layers=qformer.num_layers,
        num_heads=qformer.num_heads,
        ffn_multiplier=qformer.ffn_multiplier,
    )
    validation_qformer.load_state_dict(_detach_state_dict_to_cpu(state_dict or qformer.state_dict()))
    validation_qformer.requires_grad_(False)
    validation_qformer.eval()
    return validation_qformer


def build_detached_validation_pipeline(
    *,
    pretrained_model_name_or_path: str,
    revision: str | None,
    variant: str | None,
    torch_dtype: Any,
    transformer_lora_config: Any,
    transformer_lora_state_dict: Mapping[str, Any],
    include_text_encoder: bool = True,
    enable_model_cpu_offload: bool = False,
    logger: Any | None = None,
):
    validation_transformer = Flux2Transformer2DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer",
        revision=revision,
        variant=variant,
        torch_dtype=torch_dtype,
    )
    validation_transformer.add_adapter(transformer_lora_config)
    incompatible_keys = set_peft_model_state_dict(
        validation_transformer,
        _detach_state_dict_to_cpu(transformer_lora_state_dict),
        adapter_name="default",
    )
    unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None) if incompatible_keys is not None else None
    if unexpected_keys and logger is not None:
        logger.warning(
            "Loading detached validation LoRA weights led to unexpected keys not found in the transformer: %s",
            unexpected_keys,
        )

    validation_transformer.requires_grad_(False)
    validation_transformer.eval()

    pipeline_kwargs: dict[str, Any] = {
        "transformer": validation_transformer,
        "revision": revision,
        "variant": variant,
        "torch_dtype": torch_dtype,
    }
    if not include_text_encoder:
        pipeline_kwargs["text_encoder"] = None
        pipeline_kwargs["tokenizer"] = None

    pipeline = Flux2KleinPipeline.from_pretrained(
        pretrained_model_name_or_path,
        **pipeline_kwargs,
    )
    if enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def resolve_and_log_pilot_training(args: Any, logger: Any):
    comphoser_training = resolve_pilot_training_runtime(
        args.comphoser_mode,
        primitive_groups=args.comphoser_primitive_groups,
        qformer_num_queries=args.comphoser_qformer_num_queries,
        qformer_num_layers=args.comphoser_qformer_num_layers,
    )
    if comphoser_training.uses_prepared_pilot_dataset:
        if args.dataset_name is not None or args.instance_data_dir is not None:
            logger.info(
                "Ignoring --dataset_name/--instance_data_dir because ComPhoser pilot modes use the registered prepared dataset root."
            )
        if args.cond_image_column is not None or args.caption_column is not None or args.image_column != "image":
            logger.info("Ignoring dataset column overrides because ComPhoser pilot modes use prepared manifest metadata.")
        logger.info(
            "Resolved ComPhoser training primitive groups %s to dataset roots %s",
            comphoser_training.training_spec.controls.primitive_groups,
            comphoser_training.dataset_roots,
        )
    return comphoser_training


def build_pilot_qformer(
    transformer: Any,
    *,
    comphoser_training: Any,
    logger: Any,
) -> ComPhoserQFormer | None:
    if not comphoser_training.uses_qformer:
        return None

    qformer = ComPhoserQFormer(
        hidden_size=transformer.config.joint_attention_dim,
        cond_token_dim=transformer.config.in_channels,
        num_queries=comphoser_training.qformer_num_queries,
        num_layers=comphoser_training.qformer_num_layers,
    )
    logger.info(
        "Enabled fixed-bank ComPhoser Q-Former mode for primitive groups %s with %s query tokens and %s routing layers",
        comphoser_training.training_spec.controls.primitive_groups,
        comphoser_training.qformer_num_queries,
        comphoser_training.qformer_num_layers,
    )
    return qformer


def build_pilot_checkpoint_metadata(
    args: Any,
    *,
    train_dataset: PreparedPilotDataset | MultiPreparedPilotDataset,
    qformer: ComPhoserQFormer | None,
    comphoser_training: Any,
) -> dict[str, object] | None:
    if qformer is None:
        return None

    backbone_id = args.pretrained_model_name_or_path
    if args.revision is not None:
        backbone_id = f"{backbone_id}@{args.revision}"

    training_task_ids = tuple(
        dict.fromkeys(
            str(task_id)
            for record in train_dataset.records
            for task_id in record.task_ids
        )
    )
    training_dataset_ids = tuple(
        dict.fromkeys(str(dataset_id) for dataset_id in getattr(train_dataset, "dataset_ids", ()))
    )
    if not training_dataset_ids:
        raise ValueError("Prepared pilot checkpoint metadata requires at least one training dataset id")
    record_source = getattr(train_dataset, "record_source", None)
    if record_source is None:
        raise ValueError("Prepared pilot checkpoint metadata requires the train dataset to expose record_source")

    return build_pilot_qformer_checkpoint_metadata(
        comphoser_training.training_spec.controls.primitive_groups,
        backbone_id=backbone_id,
        qformer=qformer,
        training_task_ids=training_task_ids,
        training_dataset_ids=training_dataset_ids,
        prompt_policy_summary=build_pilot_prompt_policy_summary(
            tuple(record.prompt for record in train_dataset.records),
            source_prompts=tuple(record.source_prompt for record in train_dataset.records),
            record_source=str(record_source),
        ),
        gate_loss_weight=args.comphoser_gate_loss_weight,
        gate_loss_weight_initial=args.comphoser_gate_loss_weight_initial,
        gate_loss_weight_final=args.comphoser_gate_loss_weight_final,
        gate_loss_weight_scheduler=args.comphoser_gate_loss_weight_scheduler,
    )


def build_comphoser_validation_tracker_payload(summary: Mapping[str, Any]) -> dict[str, float | int]:
    task_id = str(summary["task_id"])
    payload: dict[str, float | int] = {}

    metrics = summary.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return payload

    psnr_summary = metrics.get("psnr_db")
    if isinstance(psnr_summary, Mapping) and psnr_summary.get("status") == "available":
        payload[f"validation/{task_id}/psnr_db_mean"] = float(psnr_summary["mean"])

    return payload


def build_unified_qformer_validation_summary(
    validation_results: Sequence[tuple[Path | str, Mapping[str, Any]]] | None,
) -> dict[str, Any]:
    distribution_sum: np.ndarray | None = None
    distribution_count = 0
    accuracy_values: list[float] = []
    loss_values: list[float] = []

    for _, summary in validation_results or ():
        samples = summary.get("samples")
        if not isinstance(samples, Sequence):
            continue
        for sample in samples:
            if not isinstance(sample, Mapping):
                continue
            outputs = sample.get("outputs")
            if not isinstance(outputs, Sequence):
                continue
            for output in outputs:
                if not isinstance(output, Mapping):
                    continue
                predicted_query_gates = output.get("predicted_query_gates")
                if isinstance(predicted_query_gates, Sequence) and len(predicted_query_gates) > 0:
                    gate_array = np.asarray(predicted_query_gates, dtype=np.float64)
                    if gate_array.ndim == 1:
                        if distribution_sum is None:
                            distribution_sum = np.zeros_like(gate_array, dtype=np.float64)
                        if gate_array.shape == distribution_sum.shape:
                            distribution_sum += gate_array
                            distribution_count += 1
                metrics = output.get("metrics")
                if not isinstance(metrics, Mapping):
                    continue
                accuracy_value = metrics.get("qformer_gate_accuracy_pct")
                if accuracy_value is not None:
                    accuracy_values.append(float(accuracy_value))
                loss_value = metrics.get("qformer_gate_loss")
                if loss_value is not None:
                    loss_values.append(float(loss_value))

    query_score_distribution = None
    if distribution_sum is not None and distribution_count > 0:
        query_score_distribution = [float(value) for value in (distribution_sum / distribution_count).tolist()]

    return {
        "status": "available" if query_score_distribution is not None else "unavailable",
        "query_score_distribution": query_score_distribution,
        "distribution_count": int(distribution_count),
        "average_accuracy_pct": (
            float(np.asarray(accuracy_values, dtype=np.float64).mean()) if accuracy_values else None
        ),
        "accuracy_count": int(len(accuracy_values)),
        "average_loss": float(np.asarray(loss_values, dtype=np.float64).mean()) if loss_values else None,
        "loss_count": int(len(loss_values)),
    }


def save_unified_qformer_validation_distribution(
    validation_results: Sequence[tuple[Path | str, Mapping[str, Any]]] | None,
    unified_summary: Mapping[str, Any],
) -> Path | None:
    distribution = unified_summary.get("query_score_distribution")
    if not isinstance(distribution, Sequence) or not distribution:
        return None
    if not validation_results:
        return None

    first_summary_path = Path(validation_results[0][0]).expanduser().resolve()
    if len(first_summary_path.parents) < 2:
        return None
    artifact_dir = first_summary_path.parents[1]
    output_path = artifact_dir / 'qformer_gate_distribution.png'

    width = 960
    height = 480
    margin_left = 64
    margin_right = 24
    margin_top = 32
    margin_bottom = 56
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    image = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    axis_color = (60, 60, 60)
    bar_color = (46, 125, 50)
    text_color = (20, 20, 20)
    draw.line((margin_left, margin_top, margin_left, margin_top + chart_height), fill=axis_color, width=2)
    draw.line(
        (margin_left, margin_top + chart_height, margin_left + chart_width, margin_top + chart_height),
        fill=axis_color,
        width=2,
    )

    max_value = max(max(float(value) for value in distribution), 1e-6)
    bar_count = len(distribution)
    slot_width = chart_width / max(bar_count, 1)
    bar_width = max(int(slot_width * 0.7), 8)

    for index, value in enumerate(distribution):
        normalized = float(value) / max_value
        bar_height = int(normalized * chart_height)
        x_center = margin_left + int((index + 0.5) * slot_width)
        x0 = x_center - (bar_width // 2)
        x1 = x0 + bar_width
        y1 = margin_top + chart_height
        y0 = y1 - bar_height
        draw.rectangle((x0, y0, x1, y1), fill=bar_color, outline=bar_color)
        label = str(index + 1)
        draw.text((x_center - 4, y1 + 8), label, fill=text_color)

    draw.text((margin_left, 8), 'Mean Q-Former Gate Score by Query', fill=text_color)
    draw.text((margin_left + 4, margin_top - 18), f'max={max_value:.3f}', fill=text_color)
    draw.text((margin_left + chart_width - 180, margin_top - 18), f'n={unified_summary.get("distribution_count", 0)}', fill=text_color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def resolve_comphoser_validation_contact_sheet_path(
    summary_path: str | Path,
    summary: Mapping[str, Any],
) -> Path | None:
    samples = summary.get("samples")
    if not isinstance(samples, Sequence) or not samples:
        return None

    first_sample = samples[0]
    if not isinstance(first_sample, Mapping):
        return None
    artifacts = first_sample.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None

    contact_sheet_relpath = artifacts.get("contact_sheet_image")
    if not isinstance(contact_sheet_relpath, str) or not contact_sheet_relpath:
        return None
    return Path(summary_path).expanduser().resolve().parent / contact_sheet_relpath


def run_comphoser_validation(
    output_dir: str | Path,
    *,
    args: Any,
    pipelines_by_mode: Mapping[str, Any],
    comphoser_training: Any,
    qformer: ComPhoserQFormer | None,
    logger: Any,
    validation_mode: str | None = None,
    artifact_subdir: str,
) -> tuple[tuple[Path, dict[str, Any]], ...] | None:
    if args.comphoser_validation_mode == "off":
        return None
    if comphoser_training.primary_task_id is None:
        raise ValueError("ComPhoser validation requires a resolved primary_task_id")

    resolved_validation_mode = validation_mode or resolve_validation_inference_mode(comphoser_training.mode)
    if args.comphoser_validation_mode == "batch":
        results = []
        for task_spec in comphoser_training.training_spec.controls.tasks:
            task_artifact_subdir = str(Path(artifact_subdir) / task_spec.dataset_id)
            summary_path, summary = save_controlled_validation_artifacts(
                output_dir,
                pipelines_by_mode=pipelines_by_mode,
                records=load_prepared_pilot_records(task_spec.dataset_root, split="val"),
                task_id=task_spec.task_id,
                qformer=qformer,
                seed=args.seed,
                validation_mode=resolved_validation_mode,
                sample_limit=args.num_validation_images,
                num_outputs_per_sample=args.num_validation_seeds_per_image,
                num_inference_steps=DEFAULT_CONTROLLED_VALIDATION_STEPS,
                guidance_scale=args.guidance_scale,
                height=args.resolution,
                width=args.resolution,
                max_sequence_length=args.max_sequence_length,
                artifact_subdir=task_artifact_subdir,
            )
            logger.info(
                "Saved ComPhoser validation artifacts for task %s to %s",
                task_spec.task_id,
                Path(summary_path).parent,
            )
            results.append((summary_path, summary))
        return tuple(results)
    else:
        prompt = args.validation_prompt or args.final_validation_prompt
        if prompt is None or args.validation_image is None:
            raise ValueError(
                "ComPhoser single validation requires --validation_image and one of "
                "--validation_prompt/--final_validation_prompt"
            )
        summary_path, summary = save_validation_artifacts(
            output_dir,
            pipelines_by_mode=pipelines_by_mode,
            cases=(
                build_single_validation_case(
                    prompt=prompt,
                    condition_image_path=args.validation_image,
                    sample_id="single_case",
                    primitive_family=(
                        comphoser_training.training_spec.controls.primitive_groups[0]
                        if comphoser_training.training_spec.controls.primitive_groups
                        else None
                    ),
                    source_task=comphoser_training.primary_task_id,
                    task_strength=1.0,
                ),
            ),
            task_id=comphoser_training.primary_task_id,
            qformer=qformer,
            seed=args.seed,
            validation_mode=resolved_validation_mode,
            num_outputs_per_sample=args.num_validation_seeds_per_image,
            num_inference_steps=DEFAULT_CONTROLLED_VALIDATION_STEPS,
            guidance_scale=args.guidance_scale,
            height=args.resolution,
            width=args.resolution,
            max_sequence_length=args.max_sequence_length,
            artifact_subdir=artifact_subdir,
        )
        logger.info("Saved ComPhoser validation artifacts to %s", Path(summary_path).parent)
        return ((summary_path, summary),)


def run_final_comphoser_export(
    args: Any,
    *,
    comphoser_training: Any,
    qformer: Any,
    qformer_state_dict: Mapping[str, Any] | None,
    comphoser_checkpoint_metadata: Mapping[str, object] | None,
    weight_dtype: Any,
    unwrap_model: Callable[[Any], Any],
    logger: Any,
    run_validation: bool = True,
) -> tuple[tuple[Path, dict[str, Any]], ...] | None:
    if qformer is None:
        if args.comphoser_mode != "baseline":
            logger.info(
                "Skipping ComPhoser controlled validation because mode '%s' does not use the controller.",
                args.comphoser_mode,
            )
        return None

    if comphoser_checkpoint_metadata is None:
        raise ValueError("Missing ComPhoser checkpoint metadata for final Q-Former export")

    qformer_to_save = unwrap_model(qformer)
    checkpoint_paths = save_pilot_qformer_checkpoint(
        args.output_dir,
        qformer=qformer_to_save,
        metadata=comphoser_checkpoint_metadata,
        state_dict=qformer_state_dict,
    )
    logger.info("Saved ComPhoser Q-Former artifacts to %s", checkpoint_paths.artifact_dir)
    if not run_validation:
        return None

    lora_pipeline = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    lora_pipeline.load_lora_weights(args.output_dir)
    lora_pipeline.enable_model_cpu_offload()
    lora_pipeline.set_progress_bar_config(disable=True)

    validation_results = run_comphoser_validation(
        args.output_dir,
        args=args,
        pipelines_by_mode={
            resolve_validation_inference_mode(comphoser_training.mode): lora_pipeline,
        },
        comphoser_training=comphoser_training,
        qformer=qformer_to_save,
        logger=logger,
        validation_mode=resolve_validation_inference_mode(comphoser_training.mode),
        artifact_subdir="controlled_validation",
    )
    if validation_results is not None:
        for controlled_validation_summary_path, controlled_validation_summary in validation_results:
            update_controlled_validation_metadata(
                controlled_validation_summary_path,
                controlled_validation_summary,
            )
            logger.info(
                "Saved ComPhoser controlled validation artifacts to %s",
                Path(controlled_validation_summary_path).parent,
            )

    del lora_pipeline
    free_memory()
    return validation_results


__all__ = [
    "build_comphoser_validation_tracker_payload",
    "build_detached_validation_pipeline",
    "build_detached_validation_qformer",
    "build_pilot_checkpoint_metadata",
    "build_pilot_qformer",
    "build_unified_qformer_validation_summary",
    "resolve_and_log_pilot_training",
    "save_unified_qformer_validation_distribution",
    "resolve_comphoser_validation_contact_sheet_path",
    "run_comphoser_validation",
    "run_final_comphoser_export",
]

"""ComPhoser-owned runtime helpers layered on top of the retained trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from peft import set_peft_model_state_dict

from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from diffusers.training_utils import free_memory

from .datasets import PreparedPilotDataset, load_prepared_pilot_records
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
    )
    if comphoser_training.uses_prepared_pilot_dataset:
        if args.dataset_name is not None or args.instance_data_dir is not None:
            logger.info(
                "Ignoring --dataset_name/--instance_data_dir because ComPhoser pilot modes use the registered prepared dataset root."
            )
        if args.cond_image_column is not None or args.caption_column is not None or args.image_column != "image":
            logger.info("Ignoring dataset column overrides because ComPhoser pilot modes use prepared manifest metadata.")
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
    )
    logger.info(
        "Enabled ComPhoser Q-Former pilot mode for task %s with %s query tokens",
        comphoser_training.primary_task_id,
        comphoser_training.qformer_num_queries,
    )
    return qformer


def build_pilot_checkpoint_metadata(
    args: Any,
    *,
    train_dataset: PreparedPilotDataset,
    qformer: ComPhoserQFormer | None,
    comphoser_training: Any,
) -> dict[str, object] | None:
    if qformer is None:
        return None

    backbone_id = args.pretrained_model_name_or_path
    if args.revision is not None:
        backbone_id = f"{backbone_id}@{args.revision}"

    return build_pilot_qformer_checkpoint_metadata(
        comphoser_training.primary_task_id,
        backbone_id=backbone_id,
        qformer=qformer,
        training_dataset_ids=(train_dataset.metadata.dataset_id,),
        prompt_policy_summary=build_pilot_prompt_policy_summary(
            tuple(record.prompt for record in train_dataset.records),
            source_prompts=tuple(record.source_prompt for record in train_dataset.records),
            record_source=train_dataset.metadata.record_source,
        ),
    )


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
) -> tuple[Path, dict[str, Any]] | None:
    if args.comphoser_validation_mode == "off":
        return None
    if comphoser_training.primary_task_id is None:
        raise ValueError("ComPhoser validation requires a resolved primary_task_id")

    resolved_validation_mode = validation_mode or resolve_validation_inference_mode(comphoser_training.mode)
    if args.comphoser_validation_mode == "batch":
        summary_path, summary = save_controlled_validation_artifacts(
            output_dir,
            pipelines_by_mode=pipelines_by_mode,
            records=load_prepared_pilot_records(comphoser_training.dataset_roots[0], split="val"),
            task_id=comphoser_training.primary_task_id,
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
            artifact_subdir=artifact_subdir,
        )
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
    return summary_path, summary


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
) -> None:
    if qformer is None:
        if args.comphoser_mode != "baseline":
            logger.info(
                "Skipping ComPhoser controlled validation because mode '%s' does not use the controller.",
                args.comphoser_mode,
            )
        return

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
        return

    lora_pipeline = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    lora_pipeline.load_lora_weights(args.output_dir)
    lora_pipeline.enable_model_cpu_offload()
    lora_pipeline.set_progress_bar_config(disable=True)

    controlled_validation_summary_path, controlled_validation_summary = save_controlled_validation_artifacts(
        args.output_dir,
        pipelines_by_mode={
            resolve_validation_inference_mode(comphoser_training.mode): lora_pipeline,
        },
        records=load_prepared_pilot_records(comphoser_training.dataset_roots[0], split="val"),
        task_id=comphoser_training.primary_task_id,
        qformer=qformer_to_save,
        seed=args.seed,
        validation_mode=resolve_validation_inference_mode(comphoser_training.mode),
        sample_limit=args.num_validation_images,
        num_outputs_per_sample=args.num_validation_seeds_per_image,
        num_inference_steps=DEFAULT_CONTROLLED_VALIDATION_STEPS,
        guidance_scale=args.guidance_scale,
        height=args.resolution,
        width=args.resolution,
        max_sequence_length=args.max_sequence_length,
    )
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


__all__ = [
    "build_detached_validation_pipeline",
    "build_detached_validation_qformer",
    "build_pilot_checkpoint_metadata",
    "build_pilot_qformer",
    "resolve_and_log_pilot_training",
    "run_comphoser_validation",
    "run_final_comphoser_export",
]

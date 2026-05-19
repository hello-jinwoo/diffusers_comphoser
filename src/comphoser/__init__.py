"""ComPhoser package."""

from __future__ import annotations

from importlib import import_module

_MODULE_EXPORTS = {
    "controls",
    "datasets",
    "evaluation",
    "inference",
    "metrics",
    "preprocessing",
    "qformer",
    "train_app",
    "train_args",
    "train_runtime",
    "training",
}

_SYMBOL_EXPORTS = {
    "ComPhoserQFormer": "comphoser.qformer",
    "DEFAULT_PRETRAINED_MODEL_NAME_OR_PATH": "comphoser.evaluation",
    "DEFAULT_QFORMER_NUM_LAYERS": "comphoser.qformer",
    "DEFAULT_QFORMER_QUERY_COUNT": "comphoser.qformer",
    "EVALUATION_MODES": "comphoser.evaluation",
    "MultiPreparedPilotDataset": "comphoser.datasets",
    "PREPARED_PRIMITIVE_FORMAT_VERSION": "comphoser.datasets",
    "PREPARED_RECORD_SOURCE_DERIVED_CONTRACT": "comphoser.datasets",
    "PREPARED_RECORD_SOURCE_MANIFEST": "comphoser.datasets",
    "PreparedPilotDataset": "comphoser.datasets",
    "PreparedPilotRecord": "comphoser.datasets",
    "PrimitiveGroupBalancedBucketBatchSampler": "comphoser.datasets",
    "QFORMER_CONTROLLER_LAYOUT_PROMPT_ROUTER_V2": "comphoser.qformer",
    "QFORMER_RUNTIME_GATE_MODE_PREDICTED_ONLY": "comphoser.qformer",
    "QFORMER_RUNTIME_GATE_MODE_TARGET_MASKED": "comphoser.qformer",
    "append_query_tokens_to_prompt": "comphoser.qformer",
    "build_batch_query_gate_target_mask": "comphoser.qformer",
    "build_controlled_validation_cases": "comphoser.inference",
    "build_controlled_validation_prompt_panel": "comphoser.inference",
    "build_dataset_from_original": "comphoser.preprocessing",
    "build_metrics_report_payload": "comphoser.evaluation",
    "build_pilot_prompt_policy_summary": "comphoser.training",
    "build_pilot_qformer_auxiliary_loss": "comphoser.training",
    "build_pilot_qformer_checkpoint_metadata": "comphoser.training",
    "build_preprocessed_dataset_from_raw": "comphoser.preprocessing",
    "build_query_gate_target_mask": "comphoser.qformer",
    "build_raw_dataset_from_original": "comphoser.preprocessing",
    "build_raw_images_from_original": "comphoser.preprocessing",
    "compute_delta_e_2000": "comphoser.metrics",
    "compute_image_metrics": "comphoser.metrics",
    "compute_psnr_db": "comphoser.metrics",
    "compute_ssim": "comphoser.metrics",
    "build_single_validation_case": "comphoser.inference",
    "collate_prepared_pilot_examples": "comphoser.datasets",
    "discover_original_paired_samples": "comphoser.preprocessing",
    "discover_raw_paired_samples": "comphoser.preprocessing",
    "evaluate_checkpoint": "comphoser.evaluation",
    "infer_dataset_name": "comphoser.preprocessing",
    "list_dataset_backed_primitive_groups": "comphoser.controls",
    "load_pilot_qformer_checkpoint": "comphoser.training",
    "load_evaluation_pipeline": "comphoser.evaluation",
    "load_evaluation_qformer": "comphoser.evaluation",
    "load_prepared_pilot_dataset_metadata": "comphoser.datasets",
    "load_prepared_pilot_records": "comphoser.datasets",
    "prepare_pilot_inference_conditioning": "comphoser.inference",
    "prepare_pilot_transformer_conditioning": "comphoser.training",
    "resolve_control_selection": "comphoser.controls",
    "resolve_evaluation_dataset": "comphoser.evaluation",
    "resolve_inference_control": "comphoser.inference",
    "resolve_evaluation_mode": "comphoser.evaluation",
    "resolve_pilot_batch_primitive_controls": "comphoser.training",
    "resolve_pilot_gate_loss_weight": "comphoser.training",
    "resolve_training_lr_scheduler_name": "comphoser.training",
    "resolve_pilot_batch_task_strengths": "comphoser.training",
    "resolve_pilot_qformer_checkpoint_paths": "comphoser.training",
    "resolve_pilot_training_runtime": "comphoser.training",
    "resolve_training_spec": "comphoser.training",
    "resolve_validation_inference_mode": "comphoser.training",
    "run_comphoser_validation": "comphoser.train_runtime",
    "save_controlled_validation_artifacts": "comphoser.inference",
    "save_pilot_qformer_checkpoint": "comphoser.training",
    "save_validation_artifacts": "comphoser.inference",
    "train_app": "comphoser.train_app",
    "update_controlled_validation_metadata": "comphoser.training",
}

__all__ = sorted(_MODULE_EXPORTS | set(_SYMBOL_EXPORTS))


def __getattr__(name: str):
    if name in _MODULE_EXPORTS:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _SYMBOL_EXPORTS:
        module = import_module(_SYMBOL_EXPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

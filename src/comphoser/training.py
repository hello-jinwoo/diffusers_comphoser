"""Training-facing helpers for ComPhoser."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from safetensors.torch import load_file as load_safetensors_file, save_file as save_safetensors_file
from torch import Tensor

from .controls import (
    PILOT_CHECKPOINT_METADATA_FIELDS,
    ResolvedPrimitiveSelection,
    build_pilot_checkpoint_metadata,
    resolve_control_selection,
)
from .datasets import PREPARED_RECORD_SOURCE_DERIVED_CONTRACT, PREPARED_RECORD_SOURCE_MANIFEST
from .qformer import ComPhoserQFormer, append_query_tokens_to_prompt

PILOT_TRAINING_MODES = ("baseline", "lora_only", "lora_qformer")
VALIDATION_INFERENCE_MODE_BY_TRAINING_MODE = {
    "baseline": "flux_only",
    "lora_only": "lora_only",
    "lora_qformer": "lora_qformer",
}
COMPHOSER_CHECKPOINT_SUBDIR = "comphoser"
COMPHOSER_SHARED_QFORMER_FILENAME = "shared_qformer.safetensors"
COMPHOSER_TASK_QUERY_BANK_FILENAME = "task_query_bank.safetensors"
COMPHOSER_METADATA_FILENAME = "metadata.json"
CONTROLLED_VALIDATION_METADATA_ARTIFACT = "controlled_validation"
_QFORMER_TASK_STATE_KEYS = frozenset({"query_bank", "task_embedding.weight"})


@dataclass(frozen=True)
class TrainingControlSpec:
    controls: ResolvedPrimitiveSelection
    checkpoint_metadata: tuple[dict[str, object], ...]

    @property
    def task_ids(self) -> tuple[str, ...]:
        return self.controls.task_ids


@dataclass(frozen=True)
class PilotTrainingRuntimeSpec:
    mode: str
    training_spec: TrainingControlSpec
    dataset_roots: tuple[str, ...]
    qformer_num_queries: int | None = None

    @property
    def uses_prepared_pilot_dataset(self) -> bool:
        return self.mode in {"lora_only", "lora_qformer"}

    @property
    def uses_qformer(self) -> bool:
        return self.mode == "lora_qformer"

    @property
    def primary_task_id(self) -> str | None:
        return self.training_spec.task_ids[0] if self.training_spec.task_ids else None


@dataclass(frozen=True)
class PilotTransformerConditioning:
    encoder_hidden_states: Tensor
    txt_ids: Tensor
    added_token_count: int
    raw_query_gates: Tensor | None = None
    query_gates: Tensor | None = None
    gate_summary: Mapping[str, Tensor] | None = None


@dataclass(frozen=True)
class PilotQFormerCheckpointPaths:
    root_dir: Path
    artifact_dir: Path
    shared_qformer_path: Path
    task_query_bank_path: Path
    metadata_path: Path


def resolve_training_spec(
    primitive_groups: Sequence[str] | str | None = None,
    task_strengths: Mapping[str, float] | Sequence[float] | float | None = None,
    metadata_overrides: Mapping[str, Mapping[str, object]] | None = None,
) -> TrainingControlSpec:
    controls = resolve_control_selection(primitive_groups=primitive_groups, task_strengths=task_strengths)
    overrides = dict(metadata_overrides or {})

    unknown_keys = set(overrides) - set(controls.task_ids)
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise KeyError(f"Metadata overrides reference unknown task ids: {unknown}")

    checkpoint_metadata = tuple(
        build_pilot_checkpoint_metadata(task.task_id, **overrides.get(task.task_id, {}))
        for task in controls.tasks
    )
    return TrainingControlSpec(controls=controls, checkpoint_metadata=checkpoint_metadata)


def resolve_pilot_training_runtime(
    run_mode: str = "baseline",
    *,
    primitive_groups: Sequence[str] | str | None = None,
    qformer_num_queries: int | None = None,
    metadata_overrides: Mapping[str, Mapping[str, object]] | None = None,
) -> PilotTrainingRuntimeSpec:
    if run_mode not in PILOT_TRAINING_MODES:
        supported = ", ".join(PILOT_TRAINING_MODES)
        raise ValueError(f"Unsupported pilot run mode '{run_mode}'. Expected one of: {supported}")

    if run_mode == "baseline":
        if primitive_groups not in (None, (), []):
            raise ValueError("primitive_groups are only valid for ComPhoser pilot modes")
        return PilotTrainingRuntimeSpec(
            mode=run_mode,
            training_spec=resolve_training_spec(metadata_overrides=metadata_overrides),
            dataset_roots=(),
            qformer_num_queries=None,
        )

    training_spec = resolve_training_spec(
        primitive_groups=primitive_groups,
        metadata_overrides=metadata_overrides,
    )
    if not training_spec.task_ids:
        raise ValueError("ComPhoser pilot modes require at least one active primitive group")
    if len(training_spec.task_ids) != 1:
        raise NotImplementedError("Stage 3B only supports a single active primitive task")

    if run_mode == "lora_qformer":
        if qformer_num_queries is None or qformer_num_queries <= 0:
            raise ValueError("lora_qformer mode requires a positive qformer_num_queries value")
    dataset_roots = tuple(task.dataset_root for task in training_spec.controls.tasks)
    return PilotTrainingRuntimeSpec(
        mode=run_mode,
        training_spec=training_spec,
        dataset_roots=dataset_roots,
        qformer_num_queries=qformer_num_queries if run_mode == "lora_qformer" else None,
    )


def resolve_pilot_batch_task_strengths(
    task_ids: Sequence[Sequence[str]],
    task_strengths: Sequence[Sequence[float]],
    *,
    expected_task_id: str,
) -> tuple[float, ...]:
    if len(task_ids) != len(task_strengths):
        raise ValueError("task_ids and task_strengths batch fields must have matching lengths")

    resolved_strengths: list[float] = []
    for sample_index, (sample_task_ids, sample_task_strengths) in enumerate(zip(task_ids, task_strengths)):
        normalized_task_ids = tuple(sample_task_ids)
        normalized_strengths = tuple(float(value) for value in sample_task_strengths)

        if len(normalized_task_ids) != len(normalized_strengths):
            raise ValueError(
                f"Sample {sample_index} has mismatched task_ids/task_strengths lengths: "
                f"{len(normalized_task_ids)} vs {len(normalized_strengths)}"
            )
        if not normalized_task_ids:
            resolved_strengths.append(0.0)
            continue
        if len(normalized_task_ids) != 1:
            raise NotImplementedError("Stage 3B only supports one task id per sample")
        if normalized_task_ids[0] != expected_task_id:
            raise ValueError(
                f"Sample {sample_index} task_id '{normalized_task_ids[0]}' does not match expected pilot task "
                f"'{expected_task_id}'"
            )

        strength = normalized_strengths[0]
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"Sample {sample_index} task strength must stay within [0.0, 1.0]")
        resolved_strengths.append(strength)

    return tuple(resolved_strengths)


def build_pilot_prompt_policy_summary(
    prompts: Sequence[str],
    *,
    source_prompts: Sequence[str | None] = (),
    record_source: str = PREPARED_RECORD_SOURCE_MANIFEST,
) -> dict[str, object]:
    unique_prompts = tuple(sorted({str(prompt) for prompt in prompts}))
    unique_source_prompts = tuple(sorted({str(prompt) for prompt in source_prompts if prompt}))
    if record_source == PREPARED_RECORD_SOURCE_MANIFEST:
        policy = "prepared_manifest_prompt"
        prompt_source = "prepared_manifest.prompt"
        notes = "Stage 3C uses prepared manifest prompts as-is; image-centric controlled validation reuses those prompts directly."
    elif record_source == PREPARED_RECORD_SOURCE_DERIVED_CONTRACT:
        policy = "contract_raw_prompt_text"
        prompt_source = "contract_dataset.raw.prompt_text"
        notes = "Stage 3C derives runtime prompts from contract dataset raw/prompt files; image-centric controlled validation reuses those prompts directly."
    else:
        raise ValueError(f"Unsupported prompt record_source '{record_source}'")
    return {
        "policy": policy,
        "prompt_source": prompt_source,
        "record_source": record_source,
        "source_prompt_field": "source_prompt",
        "unique_prompt_count": len(unique_prompts),
        "unique_source_prompt_count": len(unique_source_prompts),
        "notes": notes,
    }


def build_pilot_qformer_checkpoint_metadata(
    task_id: str,
    *,
    backbone_id: str,
    qformer: ComPhoserQFormer,
    training_dataset_ids: Sequence[str],
    prompt_policy_summary: Mapping[str, object],
    optimization_focus: str = "interpretability",
) -> dict[str, object]:
    return build_pilot_checkpoint_metadata(
        task_id,
        backbone_id=backbone_id,
        query_count=qformer.num_queries,
        query_hidden_size=qformer.hidden_size,
        training_dataset_ids=tuple(dict.fromkeys(str(dataset_id) for dataset_id in training_dataset_ids)),
        prompt_policy_summary=dict(prompt_policy_summary),
        evaluation_summary_pointers=(
            {"status": "pending", "artifact": CONTROLLED_VALIDATION_METADATA_ARTIFACT},
        ),
        baseline_comparison_pointers=(
            {
                "status": "pending",
                "modes": ("lora_qformer",),
                "artifact": CONTROLLED_VALIDATION_METADATA_ARTIFACT,
            },
        ),
        optimization_focus=optimization_focus,
        cond_token_dim=qformer.cond_token_dim,
        controller_layout="stage3c_qformer_split_v1",
    )


def resolve_pilot_qformer_checkpoint_paths(output_dir: str | Path) -> PilotQFormerCheckpointPaths:
    root_dir = Path(output_dir).expanduser().resolve()
    artifact_dir = root_dir / COMPHOSER_CHECKPOINT_SUBDIR
    return PilotQFormerCheckpointPaths(
        root_dir=root_dir,
        artifact_dir=artifact_dir,
        shared_qformer_path=artifact_dir / COMPHOSER_SHARED_QFORMER_FILENAME,
        task_query_bank_path=artifact_dir / COMPHOSER_TASK_QUERY_BANK_FILENAME,
        metadata_path=artifact_dir / COMPHOSER_METADATA_FILENAME,
    )


def has_pilot_qformer_checkpoint(output_dir: str | Path) -> bool:
    paths = resolve_pilot_qformer_checkpoint_paths(output_dir)
    return (
        paths.shared_qformer_path.is_file()
        and paths.task_query_bank_path.is_file()
        and paths.metadata_path.is_file()
    )


def save_pilot_qformer_checkpoint(
    output_dir: str | Path,
    *,
    qformer: ComPhoserQFormer,
    metadata: Mapping[str, object],
    state_dict: Mapping[str, Tensor] | None = None,
) -> PilotQFormerCheckpointPaths:
    paths = resolve_pilot_qformer_checkpoint_paths(output_dir)
    paths.artifact_dir.mkdir(parents=True, exist_ok=True)
    validate_pilot_qformer_checkpoint_metadata(metadata, qformer=qformer)

    shared_state, task_state = split_pilot_qformer_state_dict(state_dict or qformer.state_dict())
    save_safetensors_file(shared_state, str(paths.shared_qformer_path))
    save_safetensors_file(task_state, str(paths.task_query_bank_path))
    with paths.metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(metadata), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return paths


def load_pilot_qformer_checkpoint(
    input_dir: str | Path,
    *,
    qformer: ComPhoserQFormer,
    expected_task_id: str | None = None,
) -> dict[str, object]:
    paths = resolve_pilot_qformer_checkpoint_paths(input_dir)
    missing_files = [
        str(path)
        for path in (paths.shared_qformer_path, paths.task_query_bank_path, paths.metadata_path)
        if not path.is_file()
    ]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(f"Missing ComPhoser Q-Former checkpoint artifacts: {missing}")

    with paths.metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    validate_pilot_qformer_checkpoint_metadata(metadata, qformer=qformer, expected_task_id=expected_task_id)

    shared_state = dict(load_safetensors_file(str(paths.shared_qformer_path), device="cpu"))
    task_state = dict(load_safetensors_file(str(paths.task_query_bank_path), device="cpu"))
    qformer.load_state_dict({**shared_state, **task_state})
    return metadata


def validate_pilot_qformer_checkpoint_metadata(
    metadata: Mapping[str, object],
    *,
    qformer: ComPhoserQFormer | None = None,
    expected_task_id: str | None = None,
) -> None:
    missing_fields = [field for field in PILOT_CHECKPOINT_METADATA_FIELDS if field not in metadata]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"Missing required ComPhoser checkpoint metadata fields: {missing}")

    if expected_task_id is not None and metadata["task_id"] != expected_task_id:
        raise ValueError(
            f"Checkpoint metadata task_id '{metadata['task_id']}' does not match expected task_id '{expected_task_id}'"
        )
    if qformer is None:
        return

    query_count = metadata.get("query_count")
    if query_count is not None and int(query_count) != qformer.num_queries:
        raise ValueError(
            f"Checkpoint metadata query_count {query_count} does not match instantiated qformer.num_queries "
            f"{qformer.num_queries}"
        )

    query_hidden_size = metadata.get("query_hidden_size")
    if query_hidden_size is not None and int(query_hidden_size) != qformer.hidden_size:
        raise ValueError(
            f"Checkpoint metadata query_hidden_size {query_hidden_size} does not match instantiated "
            f"qformer.hidden_size {qformer.hidden_size}"
        )

    cond_token_dim = metadata.get("cond_token_dim")
    if cond_token_dim is not None and int(cond_token_dim) != qformer.cond_token_dim:
        raise ValueError(
            f"Checkpoint metadata cond_token_dim {cond_token_dim} does not match instantiated "
            f"qformer.cond_token_dim {qformer.cond_token_dim}"
        )


def update_controlled_validation_metadata(
    summary_path: str | Path,
    summary_payload: Mapping[str, object],
) -> Path:
    checkpoint_paths = resolve_pilot_qformer_checkpoint_paths(Path(summary_path).parents[2])
    if not checkpoint_paths.metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing ComPhoser metadata file for controlled validation update: {checkpoint_paths.metadata_path}"
        )

    with checkpoint_paths.metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    relative_summary_path = str(Path(summary_path).resolve().relative_to(checkpoint_paths.root_dir))
    metadata["evaluation_summary_pointers"] = [
        {
            "status": "available",
            "artifact": relative_summary_path,
            "artifact_version": summary_payload["artifact_version"],
            "active_validation_mode": summary_payload.get("active_validation_mode"),
            "case_count": summary_payload["case_count"],
            "sample_count": summary_payload.get("sample_count"),
            "run_count": summary_payload["run_count"],
            "num_validation_seeds_per_image": summary_payload.get("num_validation_seeds_per_image"),
        }
    ]
    metadata["baseline_comparison_pointers"] = [
        {
            "status": "available",
            "modes": [summary_payload["active_validation_mode"]],
            "artifact": relative_summary_path,
            "artifact_version": summary_payload["artifact_version"],
            "sample_count": summary_payload.get("sample_count"),
            "run_count": summary_payload.get("run_count"),
        }
    ]

    with checkpoint_paths.metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return checkpoint_paths.metadata_path


def resolve_validation_inference_mode(training_mode: str) -> str:
    try:
        return VALIDATION_INFERENCE_MODE_BY_TRAINING_MODE[training_mode]
    except KeyError as error:
        supported = ", ".join(PILOT_TRAINING_MODES)
        raise ValueError(f"Unsupported training mode '{training_mode}'. Expected one of: {supported}") from error


def split_pilot_qformer_state_dict(
    state_dict: Mapping[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    shared_state: dict[str, Tensor] = {}
    task_state: dict[str, Tensor] = {}

    for key, value in state_dict.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"Unexpected non-tensor value in qformer state_dict for key '{key}'")
        target = task_state if key in _QFORMER_TASK_STATE_KEYS else shared_state
        target[key] = value.detach().cpu().contiguous()

    missing_task_keys = sorted(_QFORMER_TASK_STATE_KEYS - set(task_state))
    if missing_task_keys:
        missing = ", ".join(missing_task_keys)
        raise ValueError(f"Q-Former state_dict is missing required task-owned keys: {missing}")

    return shared_state, task_state


def prepare_pilot_transformer_conditioning(
    prompt_embeds: Tensor,
    txt_ids: Tensor,
    cond_tokens: Tensor,
    *,
    qformer: ComPhoserQFormer | None,
    task_id: str | None = None,
    task_strengths: Sequence[float] | Tensor | None = None,
) -> PilotTransformerConditioning:
    if qformer is None:
        return PilotTransformerConditioning(
            encoder_hidden_states=prompt_embeds,
            txt_ids=txt_ids,
            added_token_count=0,
        )
    if task_id is None:
        raise ValueError("task_id is required when qformer conditioning is enabled")

    first_param = next(qformer.parameters(), None)
    if first_param is None:
        raise ValueError("qformer must have at least one parameter")

    controller_output = qformer(
        prompt_embeds.to(device=first_param.device, dtype=first_param.dtype),
        cond_tokens.to(device=first_param.device, dtype=first_param.dtype),
        task_ids=(task_id,),
        task_strengths=task_strengths,
    )
    augmented = append_query_tokens_to_prompt(
        prompt_embeds,
        txt_ids,
        controller_output.query_group.to(device=prompt_embeds.device),
    )
    return PilotTransformerConditioning(
        encoder_hidden_states=augmented.encoder_hidden_states,
        txt_ids=augmented.txt_ids,
        added_token_count=augmented.added_token_count,
        raw_query_gates=controller_output.raw_query_gates.to(device=prompt_embeds.device),
        query_gates=controller_output.query_gates.to(device=prompt_embeds.device),
        gate_summary={
            key: value.to(device=prompt_embeds.device) for key, value in controller_output.gate_summary.items()
        },
    )
__all__ = [
    "CONTROLLED_VALIDATION_METADATA_ARTIFACT",
    "COMPHOSER_CHECKPOINT_SUBDIR",
    "COMPHOSER_METADATA_FILENAME",
    "COMPHOSER_SHARED_QFORMER_FILENAME",
    "COMPHOSER_TASK_QUERY_BANK_FILENAME",
    "PILOT_TRAINING_MODES",
    "VALIDATION_INFERENCE_MODE_BY_TRAINING_MODE",
    "PilotQFormerCheckpointPaths",
    "PilotTrainingRuntimeSpec",
    "PilotTransformerConditioning",
    "TrainingControlSpec",
    "build_pilot_prompt_policy_summary",
    "build_pilot_qformer_checkpoint_metadata",
    "has_pilot_qformer_checkpoint",
    "load_pilot_qformer_checkpoint",
    "prepare_pilot_transformer_conditioning",
    "resolve_pilot_batch_task_strengths",
    "resolve_pilot_qformer_checkpoint_paths",
    "resolve_pilot_training_runtime",
    "resolve_training_spec",
    "resolve_validation_inference_mode",
    "save_pilot_qformer_checkpoint",
    "split_pilot_qformer_state_dict",
    "update_controlled_validation_metadata",
    "validate_pilot_qformer_checkpoint_metadata",
]

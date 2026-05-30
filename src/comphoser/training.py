"""Training-facing helpers for ComPhoser."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from safetensors.torch import load_file as load_safetensors_file, save_file as save_safetensors_file
from torch import Tensor
from torch.nn import functional as F

from .controls import (
    PILOT_CHECKPOINT_METADATA_FIELDS,
    PILOT_CHECKPOINT_METADATA_VERSION,
    PILOT_PRIMITIVE_FAMILY_ORDER,
    PrimitiveTaskSpec,
    ResolvedPrimitiveSelection,
    build_pilot_checkpoint_metadata,
    get_task_spec,
    get_task_specs_for_primitive_group,
    normalize_primitive_groups,
    resolve_control_selection,
    resolve_dataset_task_spec_by_id,
    resolve_primitive_group,
)
from .datasets import PREPARED_RECORD_SOURCE_DERIVED_CONTRACT, PREPARED_RECORD_SOURCE_MANIFEST
from .qformer import (
    DEFAULT_QFORMER_NUM_LAYERS,
    DEFAULT_QFORMER_QUERY_COUNT,
    ComPhoserQFormer,
    QFORMER_CONTROLLER_LAYOUT_PROMPT_ROUTER_V2,
    append_query_tokens_to_prompt,
)

PILOT_GATE_LOSS_WEIGHT_SCHEDULERS = ("linear", "logarithmic", "exponential")
PILOT_TRAINING_MODES = ("baseline", "lora_only", "lora_qformer")
VALIDATION_INFERENCE_MODE_BY_TRAINING_MODE = {
    "baseline": "flux_only",
    "lora_only": "lora_only",
    "lora_qformer": "lora_qformer",
}
COMPHOSER_CHECKPOINT_SUBDIR = "comphoser"
COMPHOSER_SHARED_QFORMER_FILENAME = "shared_qwp_or_qformer.safetensors"
COMPHOSER_GLOBAL_QUERY_BANK_FILENAME = "global_query_bank.safetensors"
COMPHOSER_TASK_QUERY_BANK_FILENAME = COMPHOSER_GLOBAL_QUERY_BANK_FILENAME
COMPHOSER_METADATA_FILENAME = "metadata.json"
CONTROLLED_VALIDATION_METADATA_ARTIFACT = "controlled_validation"
_QFORMER_QUERY_BANK_STATE_KEYS = frozenset({"query_bank"})


@dataclass(frozen=True)
class TrainingControlSpec:
    controls: ResolvedPrimitiveSelection
    checkpoint_metadata: dict[str, object]

    @property
    def task_ids(self) -> tuple[str, ...]:
        return self.controls.task_ids


@dataclass(frozen=True)
class PilotTrainingRuntimeSpec:
    mode: str
    training_spec: TrainingControlSpec
    dataset_roots: tuple[str, ...]
    training_task_specs: tuple[PrimitiveTaskSpec, ...] = ()
    qformer_num_queries: int | None = None
    qformer_num_layers: int | None = None

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
class PilotBatchPrimitiveControls:
    primitive_groups: tuple[tuple[str, ...], ...]
    primitive_strengths: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class PilotTransformerConditioning:
    encoder_hidden_states: Tensor
    txt_ids: Tensor
    added_token_count: int
    gate_targets: Tensor | None = None
    raw_query_gates: Tensor | None = None
    predicted_query_gates: Tensor | None = None
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

    valid_override_keys = set(controls.task_ids) | set(controls.primitive_groups) | {"__all__"}
    unknown_keys = set(overrides) - valid_override_keys
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise KeyError(f"Metadata overrides reference unknown primitive groups or task ids: {unknown}")

    merged_overrides: dict[str, object] = {}
    merged_overrides.update(overrides.get("__all__", {}))
    for key in controls.primitive_groups:
        merged_overrides.update(overrides.get(key, {}))
    for key in controls.task_ids:
        merged_overrides.update(overrides.get(key, {}))

    checkpoint_metadata = build_pilot_checkpoint_metadata(
        controls.primitive_groups,
        training_task_ids=controls.task_ids,
        **merged_overrides,
    )
    return TrainingControlSpec(controls=controls, checkpoint_metadata=checkpoint_metadata)


def resolve_pilot_training_runtime(
    run_mode: str = "baseline",
    *,
    primitive_groups: Sequence[str] | str | None = None,
    qformer_num_queries: int | None = None,
    qformer_num_layers: int | None = None,
    metadata_overrides: Mapping[str, Mapping[str, object]] | None = None,
    exclude_dataset_ids: Sequence[str] | None = None,
    train_dataset_ids: Sequence[str] | None = None,
    include_non_catalog_in_discovery: bool = False,
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
            qformer_num_layers=None,
        )

    training_spec = resolve_training_spec(
        primitive_groups=primitive_groups,
        metadata_overrides=metadata_overrides,
    )
    if not training_spec.controls.primitive_groups:
        raise ValueError("ComPhoser pilot modes require at least one active primitive group")

    dataset_tasks: list[PrimitiveTaskSpec] = []
    if train_dataset_ids:
        # Explicit allow-list: bypass the primitive_groups filter and the
        # exclude_dataset_ids subtraction entirely. Each name is resolved either to a
        # discovered catalog task or to a synthetic non-catalog spec via the contract
        # folder under data/.
        requested = list(dict.fromkeys(str(name) for name in train_dataset_ids if name))
        if not requested:
            raise ValueError("--train_dataset_ids was provided but resolved to an empty list")
        for name in requested:
            dataset_tasks.append(resolve_dataset_task_spec_by_id(name))
        if exclude_dataset_ids:
            warnings.warn(
                "--exclude_dataset_ids is ignored when --train_dataset_ids is set "
                "(the explicit list is treated as exhaustive).",
                stacklevel=2,
            )
    else:
        missing_groups: list[str] = []
        for primitive_group in training_spec.controls.primitive_groups:
            group_tasks = tuple(
                task for task in get_task_specs_for_primitive_group(primitive_group) if task.is_dataset_ready
            )
            if not group_tasks:
                missing_groups.append(primitive_group)
                continue
            dataset_tasks.extend(group_tasks)

        if missing_groups:
            missing = ", ".join(missing_groups)
            raise NotImplementedError(
                f"ComPhoser training does not have dataset-backed task routing for primitive groups: {missing}"
            )

        if include_non_catalog_in_discovery:
            # All-in-one mode: the catalog gate is not utilized on the supervision side
            # (no per-family BCE loss), so surface non-catalog folders too (e.g. downstream_*).
            # They enter with primitive_group="" and the trainer's existing no-family handling
            # produces an all-zero gate_targets mask for those samples.
            from .controls import discover_dataset_task_specs as _discover_all

            seen = {task.dataset_id for task in dataset_tasks}
            for spec in _discover_all(include_non_catalog=True):
                if not spec.is_dataset_ready:
                    continue
                if spec.primitive_group:
                    continue  # catalog spec, already collected above
                if spec.dataset_id in seen:
                    continue
                dataset_tasks.append(spec)
                seen.add(spec.dataset_id)

    if exclude_dataset_ids and not train_dataset_ids:
        excluded = {str(name) for name in exclude_dataset_ids if name}
        if excluded:
            pre_ids = {task.dataset_id for task in dataset_tasks}
            unmatched = excluded - pre_ids
            if unmatched:
                warnings.warn(
                    "--exclude_dataset_ids contains entries that did not match any discovered "
                    f"dataset for the selected primitive_groups: {sorted(unmatched)}",
                    stacklevel=2,
                )
            dataset_tasks = [task for task in dataset_tasks if task.dataset_id not in excluded]
            if not dataset_tasks:
                raise ValueError(
                    "--exclude_dataset_ids removed every discovered dataset for the selected "
                    "primitive_groups; nothing left to train on"
                )

    if run_mode == "lora_qformer":
        if qformer_num_queries is None or qformer_num_queries <= 0:
            raise ValueError("lora_qformer mode requires a positive qformer_num_queries value")
        if qformer_num_queries != DEFAULT_QFORMER_QUERY_COUNT:
            raise ValueError(
                f"lora_qformer mode uses a fixed global query bank of {DEFAULT_QFORMER_QUERY_COUNT} tokens"
            )
        resolved_num_layers = DEFAULT_QFORMER_NUM_LAYERS if qformer_num_layers is None else qformer_num_layers
        if resolved_num_layers <= 0:
            raise ValueError("lora_qformer mode requires a positive qformer_num_layers value")
    else:
        resolved_num_layers = None

    dataset_roots = tuple(dict.fromkeys(task.dataset_root for task in dataset_tasks))
    # Preserve the actual training-task list (catalog + any non-catalog folders surfaced via
    # include_non_catalog_in_discovery or --train_dataset_ids). Validation fan-out reads this
    # so it covers exactly what was trained; controls.tasks is catalog-only by design.
    seen_dataset_ids: set[str] = set()
    deduped_training_tasks: list[PrimitiveTaskSpec] = []
    for task in dataset_tasks:
        if task.dataset_id in seen_dataset_ids:
            continue
        seen_dataset_ids.add(task.dataset_id)
        deduped_training_tasks.append(task)
    return PilotTrainingRuntimeSpec(
        mode=run_mode,
        training_spec=training_spec,
        dataset_roots=dataset_roots,
        training_task_specs=tuple(deduped_training_tasks),
        qformer_num_queries=qformer_num_queries if run_mode == "lora_qformer" else None,
        qformer_num_layers=resolved_num_layers,
    )


def resolve_pilot_batch_primitive_controls(
    task_ids: Sequence[Sequence[str]],
    task_strengths: Sequence[Sequence[float]],
) -> PilotBatchPrimitiveControls:
    if len(task_ids) != len(task_strengths):
        raise ValueError("task_ids and task_strengths batch fields must have matching lengths")

    batch_groups: list[tuple[str, ...]] = []
    batch_strengths: list[tuple[float, ...]] = []
    for sample_index, (sample_task_ids, sample_task_strengths) in enumerate(zip(task_ids, task_strengths)):
        normalized_task_ids = tuple(sample_task_ids)
        normalized_strengths = tuple(float(value) for value in sample_task_strengths)

        if len(normalized_task_ids) != len(normalized_strengths):
            raise ValueError(
                f"Sample {sample_index} has mismatched task_ids/task_strengths lengths: "
                f"{len(normalized_task_ids)} vs {len(normalized_strengths)}"
            )
        if not normalized_task_ids:
            batch_groups.append(())
            batch_strengths.append(())
            continue

        grouped_strengths: dict[str, float] = {}
        for task_id, strength in zip(normalized_task_ids, normalized_strengths):
            if not 0.0 <= strength <= 1.0:
                raise ValueError(f"Sample {sample_index} task strength must stay within [0.0, 1.0]")
            try:
                task_spec = get_task_spec(task_id)
            except KeyError:
                # Non-catalog task_id (e.g. downstream_isp__ZRR loaded via the direct-load
                # fallback for --downstream_target_dataset_id). Treat as "no active family" —
                # the QFormer produces an all-zero gate_targets mask for this sample. The
                # strategies that skip the BCE (step_by_step_stage1/3, all_in_one) never consume
                # it; the strategies that DO compute it (step_by_step_stage2, single_dataset,
                # legacy None) are expected to train on catalog folders, so an all-zero target
                # here only arises from a non-catalog folder under one of those strategies (e.g.
                # single_dataset on a downstream_* folder), where it pushes every gate toward 0.
                continue
            grouped_strengths[task_spec.primitive_group] = max(grouped_strengths.get(task_spec.primitive_group, 0.0), strength)

        if len(grouped_strengths) > 1:
            resolved_groups = ", ".join(sorted(grouped_strengths))
            raise NotImplementedError(
                "v1 auxiliary gate supervision expects at most one primitive family per sample; "
                f"sample {sample_index} resolved to: {resolved_groups}"
            )

        ordered_groups = normalize_primitive_groups(tuple(grouped_strengths))
        batch_groups.append(ordered_groups)
        batch_strengths.append(tuple(grouped_strengths[group] for group in ordered_groups))

    return PilotBatchPrimitiveControls(
        primitive_groups=tuple(batch_groups),
        primitive_strengths=tuple(batch_strengths),
    )


def resolve_pilot_batch_task_strengths(
    task_ids: Sequence[Sequence[str]],
    task_strengths: Sequence[Sequence[float]],
    *,
    expected_task_id: str,
) -> tuple[float, ...]:
    expected_group = get_task_spec(expected_task_id).primitive_group
    batch_controls = resolve_pilot_batch_primitive_controls(task_ids, task_strengths)

    resolved_strengths: list[float] = []
    for sample_index, (sample_groups, sample_strengths) in enumerate(
        zip(batch_controls.primitive_groups, batch_controls.primitive_strengths)
    ):
        if not sample_groups:
            resolved_strengths.append(0.0)
            continue
        if len(sample_groups) != 1 or sample_groups[0] != expected_group:
            raise ValueError(
                f"Sample {sample_index} resolved to primitive groups {sample_groups}, "
                f"which do not match expected pilot task '{expected_task_id}'"
            )
        resolved_strengths.append(sample_strengths[0])

    return tuple(resolved_strengths)


def build_pilot_qformer_auxiliary_loss(
    raw_query_gates: Tensor,
    gate_targets: Tensor,
) -> Tensor:
    if raw_query_gates.shape != gate_targets.shape:
        raise ValueError(
            f"raw_query_gates and gate_targets must share the same shape, received "
            f"{tuple(raw_query_gates.shape)} vs {tuple(gate_targets.shape)}"
        )
    return F.binary_cross_entropy_with_logits(raw_query_gates.float(), gate_targets.float())


_GATE_LOSS_SKIPPING_STRATEGIES = frozenset({"step_by_step_stage1", "all_in_one", "step_by_step_stage3"})


def should_skip_gate_loss(training_strategy: str | None) -> bool:
    """Whether the BCE auxiliary gate loss should be skipped for this training strategy.

    Stage 1 trains identity with no per-sample family target, all-in-one treats every task
    equally without a family concept, and Stage 3 has a frozen Q-Former (any BCE here would
    just be no-op gradients on frozen params). Other strategies — None (legacy), Stage 2,
    single_dataset — compute the BCE gate loss normally.
    """
    return training_strategy in _GATE_LOSS_SKIPPING_STRATEGIES


def resolve_pilot_gate_loss_weight(
    initial_weight: float,
    final_weight: float,
    scheduler: str,
    *,
    current_step: int,
    total_steps: int,
) -> float:
    initial = float(initial_weight)
    final = float(final_weight)

    if scheduler not in PILOT_GATE_LOSS_WEIGHT_SCHEDULERS:
        supported = ", ".join(PILOT_GATE_LOSS_WEIGHT_SCHEDULERS)
        raise ValueError(f"Unsupported gate-loss weight scheduler {scheduler!r}. Expected one of: {supported}")

    if total_steps <= 1:
        return initial

    last_step_index = total_steps - 1
    step_index = min(max(int(current_step), 0), last_step_index)
    progress = step_index / last_step_index

    if scheduler == "linear":
        return initial + ((final - initial) * progress)

    if scheduler == "logarithmic":
        if initial <= 0.0 or final <= 0.0:
            raise ValueError("logarithmic gate-loss weight scheduling requires positive endpoint values")
        return math.exp(((1.0 - progress) * math.log(initial)) + (progress * math.log(final)))

    scaled_progress = math.expm1(progress) / math.expm1(1.0)
    return initial + ((final - initial) * scaled_progress)


def resolve_training_lr_scheduler_name(scheduler_name: str, *, lr_warmup_steps: int) -> str:
    normalized_name = str(scheduler_name)
    if normalized_name == "constant" and int(lr_warmup_steps) > 0:
        return "constant_with_warmup"
    return normalized_name


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
        notes = "ComPhoser uses prepared manifest prompts as-is; controlled validation reuses those prompts directly."
    elif record_source == PREPARED_RECORD_SOURCE_DERIVED_CONTRACT:
        policy = "contract_raw_prompt_text"
        prompt_source = "contract_dataset.raw.prompt_text"
        notes = "ComPhoser derives runtime prompts from contract dataset raw/prompt files; controlled validation reuses them directly."
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
    primitive_groups: Sequence[str] | str | None,
    *,
    backbone_id: str,
    qformer: ComPhoserQFormer,
    training_task_ids: Sequence[str],
    training_dataset_ids: Sequence[str],
    prompt_policy_summary: Mapping[str, object],
    optimization_focus: str = "interpretability",
    gate_loss_weight: float = 0.1,
    gate_loss_weight_initial: float | None = None,
    gate_loss_weight_final: float | None = None,
    gate_loss_weight_scheduler: str = "linear",
    extra_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    resolved_initial_weight = float(gate_loss_weight if gate_loss_weight_initial is None else gate_loss_weight_initial)
    resolved_final_weight = float(gate_loss_weight if gate_loss_weight_final is None else gate_loss_weight_final)

    return build_pilot_checkpoint_metadata(
        primitive_groups,
        training_task_ids=tuple(dict.fromkeys(str(task_id) for task_id in training_task_ids)),
        backbone_id=backbone_id,
        query_count=qformer.num_queries,
        query_hidden_size=qformer.hidden_size,
        queries_per_primitive=qformer.queries_per_primitive,
        num_layers=qformer.num_layers,
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
        gate_loss_weight=resolved_initial_weight,
        gate_loss_weight_initial=resolved_initial_weight,
        gate_loss_weight_final=resolved_final_weight,
        gate_loss_weight_scheduler=str(gate_loss_weight_scheduler),
        cond_token_dim=qformer.cond_token_dim,
        controller_layout=QFORMER_CONTROLLER_LAYOUT_PROMPT_ROUTER_V2,
        routing_context="prompt_only",
        # Run-provenance fields (training_strategy / sampling_policy / seed). Recorded so a
        # checkpoint alone identifies which strategy + config produced it (R13). These are
        # additive — not in PILOT_CHECKPOINT_METADATA_FIELDS — so legacy checkpoints still load.
        **{str(key): value for key, value in dict(extra_metadata or {}).items()},
    )


def resolve_pilot_qformer_checkpoint_paths(output_dir: str | Path) -> PilotQFormerCheckpointPaths:
    root_dir = Path(output_dir).expanduser().resolve()
    artifact_dir = root_dir / COMPHOSER_CHECKPOINT_SUBDIR
    return PilotQFormerCheckpointPaths(
        root_dir=root_dir,
        artifact_dir=artifact_dir,
        shared_qformer_path=artifact_dir / COMPHOSER_SHARED_QFORMER_FILENAME,
        task_query_bank_path=artifact_dir / COMPHOSER_GLOBAL_QUERY_BANK_FILENAME,
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

    shared_state, query_bank_state = split_pilot_qformer_state_dict(state_dict or qformer.state_dict())
    save_safetensors_file(shared_state, str(paths.shared_qformer_path))
    save_safetensors_file(query_bank_state, str(paths.task_query_bank_path))
    with paths.metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(metadata), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return paths


def load_pilot_qformer_checkpoint(
    input_dir: str | Path,
    *,
    qformer: ComPhoserQFormer,
    expected_primitive_groups: Sequence[str] | str | None = None,
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
    validate_pilot_qformer_checkpoint_metadata(
        metadata,
        qformer=qformer,
        expected_primitive_groups=expected_primitive_groups,
    )

    shared_state = dict(load_safetensors_file(str(paths.shared_qformer_path), device="cpu"))
    query_bank_state = dict(load_safetensors_file(str(paths.task_query_bank_path), device="cpu"))
    qformer.load_state_dict({**shared_state, **query_bank_state})
    return metadata


def validate_pilot_qformer_checkpoint_metadata(
    metadata: Mapping[str, object],
    *,
    qformer: ComPhoserQFormer | None = None,
    expected_primitive_groups: Sequence[str] | str | None = None,
) -> None:
    missing_fields = [field for field in PILOT_CHECKPOINT_METADATA_FIELDS if field not in metadata]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"Missing required ComPhoser checkpoint metadata fields: {missing}")

    metadata_version = str(metadata["metadata_version"])
    if metadata_version != PILOT_CHECKPOINT_METADATA_VERSION:
        raise ValueError(
            f"Checkpoint metadata_version {metadata_version!r} does not match "
            f"expected fixed-bank version {PILOT_CHECKPOINT_METADATA_VERSION!r}"
        )

    primitive_family_order = tuple(
        resolve_primitive_group(str(group)).primitive_group for group in metadata["primitive_family_order"]
    )
    if primitive_family_order != PILOT_PRIMITIVE_FAMILY_ORDER:
        raise ValueError(
            f"Checkpoint metadata primitive_family_order {primitive_family_order} does not match "
            f"expected fixed-bank order {PILOT_PRIMITIVE_FAMILY_ORDER}"
        )

    if expected_primitive_groups is not None:
        expected_groups = normalize_primitive_groups(expected_primitive_groups)
        actual_groups = normalize_primitive_groups(metadata["primitive_groups"])
        if actual_groups != expected_groups:
            raise ValueError(
                f"Checkpoint metadata primitive_groups {actual_groups} do not match expected {expected_groups}"
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

    queries_per_primitive = metadata.get("queries_per_primitive")
    if queries_per_primitive is not None and int(queries_per_primitive) != qformer.queries_per_primitive:
        raise ValueError(
            f"Checkpoint metadata queries_per_primitive {queries_per_primitive} does not match instantiated "
            f"qformer.queries_per_primitive {qformer.queries_per_primitive}"
        )

    cond_token_dim = metadata.get("cond_token_dim")
    if cond_token_dim is not None and int(cond_token_dim) != qformer.cond_token_dim:
        raise ValueError(
            f"Checkpoint metadata cond_token_dim {cond_token_dim} does not match instantiated "
            f"qformer.cond_token_dim {qformer.cond_token_dim}"
        )

    num_layers = metadata.get("num_layers")
    resolved_num_layers = DEFAULT_QFORMER_NUM_LAYERS if num_layers is None else int(num_layers)
    if resolved_num_layers != qformer.num_layers:
        raise ValueError(
            f"Checkpoint metadata num_layers {resolved_num_layers} does not match instantiated "
            f"qformer.num_layers {qformer.num_layers}"
        )


def update_controlled_validation_metadata(
    summary_path: str | Path,
    summary_payload: Mapping[str, object],
) -> Path:
    resolved_summary_path = Path(summary_path).expanduser().resolve()
    checkpoint_root = None
    for ancestor in resolved_summary_path.parents:
        if ancestor.name == COMPHOSER_CHECKPOINT_SUBDIR:
            checkpoint_root = ancestor.parent
            break
    if checkpoint_root is None:
        raise ValueError(
            f"Controlled validation summary path must be under a '{COMPHOSER_CHECKPOINT_SUBDIR}/' artifact root: "
            f"{resolved_summary_path}"
        )

    checkpoint_paths = resolve_pilot_qformer_checkpoint_paths(checkpoint_root)
    if not checkpoint_paths.metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing ComPhoser metadata file for controlled validation update: {checkpoint_paths.metadata_path}"
        )

    with checkpoint_paths.metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    relative_summary_path = str(resolved_summary_path.relative_to(checkpoint_paths.root_dir))
    evaluation_pointer = {
        "status": "available",
        "artifact": relative_summary_path,
        "artifact_version": summary_payload["artifact_version"],
        "active_validation_mode": summary_payload.get("active_validation_mode"),
        "case_count": summary_payload["case_count"],
        "sample_count": summary_payload.get("sample_count"),
        "run_count": summary_payload["run_count"],
        "num_validation_seeds_per_image": summary_payload.get("num_validation_seeds_per_image"),
    }
    baseline_pointer = {
        "status": "available",
        "modes": [summary_payload["active_validation_mode"]],
        "artifact": relative_summary_path,
        "artifact_version": summary_payload["artifact_version"],
        "sample_count": summary_payload.get("sample_count"),
        "run_count": summary_payload.get("run_count"),
    }
    placeholder_artifacts = {
        CONTROLLED_VALIDATION_METADATA_ARTIFACT,
        f"{COMPHOSER_CHECKPOINT_SUBDIR}/{CONTROLLED_VALIDATION_METADATA_ARTIFACT}",
    }
    metadata["evaluation_summary_pointers"] = [
        pointer
        for pointer in metadata.get("evaluation_summary_pointers", [])
        if pointer.get("artifact") not in placeholder_artifacts and pointer.get("artifact") != relative_summary_path
    ] + [evaluation_pointer]
    metadata["baseline_comparison_pointers"] = [
        pointer
        for pointer in metadata.get("baseline_comparison_pointers", [])
        if pointer.get("artifact") not in placeholder_artifacts and pointer.get("artifact") != relative_summary_path
    ] + [baseline_pointer]

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


VALIDATION_MODEL_CPU_OFFLOAD_POLICIES = ("auto", "on", "off")
# Threshold for the `auto` policy: when total device VRAM is at least this many GiB, we skip
# diffusers' enable_model_cpu_offload() and keep the validation pipeline GPU-resident. FLUX.2
# Klein 4B at bf16 peaks around 27 GiB during inference; 48 GiB leaves comfortable headroom
# for the training-side transformer that stays resident on the same device during periodic
# validation. Smaller GPUs fall back to offload to avoid OOM.
VALIDATION_AUTO_GPU_RESIDENT_VRAM_GIB = 48


def resolve_validation_enable_model_cpu_offload(
    policy: str,
    device: torch.device | str | None,
) -> bool:
    """Resolve the validation-pipeline CPU-offload decision for ``policy`` on ``device``.

    Returns ``True`` if `enable_model_cpu_offload()` should be applied to the validation
    pipeline, ``False`` if the pipeline should be kept GPU-resident.
    """

    if policy not in VALIDATION_MODEL_CPU_OFFLOAD_POLICIES:
        supported = ", ".join(VALIDATION_MODEL_CPU_OFFLOAD_POLICIES)
        raise ValueError(
            f"Unsupported validation_model_cpu_offload policy '{policy}'. Expected one of: {supported}"
        )
    if policy == "on":
        return True
    if policy == "off":
        return False
    # auto
    if device is None:
        return True
    resolved = torch.device(device) if not isinstance(device, torch.device) else device
    if resolved.type != "cuda":
        return False
    properties = torch.cuda.get_device_properties(resolved)
    total_gib = properties.total_memory / (1024**3)
    return total_gib < VALIDATION_AUTO_GPU_RESIDENT_VRAM_GIB


def split_pilot_qformer_state_dict(
    state_dict: Mapping[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    shared_state: dict[str, Tensor] = {}
    query_bank_state: dict[str, Tensor] = {}

    for key, value in state_dict.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"Unexpected non-tensor value in qformer state_dict for key '{key}'")
        target = query_bank_state if key in _QFORMER_QUERY_BANK_STATE_KEYS else shared_state
        target[key] = value.detach().cpu().contiguous()

    missing_query_bank_keys = sorted(_QFORMER_QUERY_BANK_STATE_KEYS - set(query_bank_state))
    if missing_query_bank_keys:
        missing = ", ".join(missing_query_bank_keys)
        raise ValueError(f"Q-Former state_dict is missing required query-bank keys: {missing}")

    return shared_state, query_bank_state


def prepare_pilot_transformer_conditioning(
    prompt_embeds: Tensor,
    txt_ids: Tensor,
    cond_tokens: Tensor,
    *,
    qformer: ComPhoserQFormer | None,
    primitive_groups: Sequence[Sequence[str] | str] | Sequence[str] | str | None = None,
    primitive_strengths: Sequence[Sequence[float] | float] | Sequence[float] | Mapping[str, float] | Tensor | float | None = None,
    explicit_token_masking: Sequence[float] | Tensor | None = None,
) -> PilotTransformerConditioning:
    if qformer is None:
        return PilotTransformerConditioning(
            encoder_hidden_states=prompt_embeds,
            txt_ids=txt_ids,
            added_token_count=0,
        )

    first_param = next(qformer.parameters(), None)
    if first_param is None:
        raise ValueError("qformer must have at least one parameter")

    controller_output = qformer(
        prompt_embeds.to(device=first_param.device, dtype=first_param.dtype),
        cond_tokens.to(device=first_param.device, dtype=first_param.dtype),
        primitive_groups=primitive_groups,
        primitive_strengths=primitive_strengths,
        explicit_token_masking=explicit_token_masking,
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
        gate_targets=controller_output.gate_targets.to(device=prompt_embeds.device),
        raw_query_gates=controller_output.raw_query_gates.to(device=prompt_embeds.device),
        predicted_query_gates=controller_output.predicted_query_gates.to(device=prompt_embeds.device),
        query_gates=controller_output.query_gates.to(device=prompt_embeds.device),
        gate_summary={
            key: value.to(device=prompt_embeds.device) for key, value in controller_output.gate_summary.items()
        },
    )


__all__ = [
    "CONTROLLED_VALIDATION_METADATA_ARTIFACT",
    "COMPHOSER_CHECKPOINT_SUBDIR",
    "COMPHOSER_GLOBAL_QUERY_BANK_FILENAME",
    "COMPHOSER_METADATA_FILENAME",
    "COMPHOSER_SHARED_QFORMER_FILENAME",
    "COMPHOSER_TASK_QUERY_BANK_FILENAME",
    "PILOT_GATE_LOSS_WEIGHT_SCHEDULERS",
    "PILOT_TRAINING_MODES",
    "QFORMER_CONTROLLER_LAYOUT_PROMPT_ROUTER_V2",
    "VALIDATION_AUTO_GPU_RESIDENT_VRAM_GIB",
    "VALIDATION_INFERENCE_MODE_BY_TRAINING_MODE",
    "VALIDATION_MODEL_CPU_OFFLOAD_POLICIES",
    "PilotBatchPrimitiveControls",
    "PilotQFormerCheckpointPaths",
    "PilotTrainingRuntimeSpec",
    "PilotTransformerConditioning",
    "TrainingControlSpec",
    "build_pilot_prompt_policy_summary",
    "build_pilot_qformer_auxiliary_loss",
    "resolve_pilot_gate_loss_weight",
    "resolve_training_lr_scheduler_name",
    "build_pilot_qformer_checkpoint_metadata",
    "has_pilot_qformer_checkpoint",
    "load_pilot_qformer_checkpoint",
    "prepare_pilot_transformer_conditioning",
    "resolve_pilot_batch_primitive_controls",
    "resolve_pilot_batch_task_strengths",
    "resolve_pilot_qformer_checkpoint_paths",
    "resolve_pilot_training_runtime",
    "resolve_training_spec",
    "resolve_validation_enable_model_cpu_offload",
    "resolve_validation_inference_mode",
    "save_pilot_qformer_checkpoint",
    "should_skip_gate_loss",
    "split_pilot_qformer_state_dict",
    "update_controlled_validation_metadata",
    "validate_pilot_qformer_checkpoint_metadata",
]

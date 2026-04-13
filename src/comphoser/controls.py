"""Pilot control registry for ComPhoser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

PILOT_PRIMITIVE_FAMILY_ORDER = ("tone_color", "exposure", "detail", "dof")
PILOT_CHECKPOINT_METADATA_VERSION = "comphoser-qformer-pilot-v1"
DEFAULT_PRIMITIVE_GROUP = "detail"
PILOT_CHECKPOINT_METADATA_FIELDS = (
    "metadata_version",
    "task_id",
    "primitive_group",
    "backbone_id",
    "query_count",
    "query_hidden_size",
    "training_dataset_ids",
    "prompt_policy_summary",
    "evaluation_summary_pointers",
    "baseline_comparison_pointers",
    "optimization_focus",
)


@dataclass(frozen=True)
class PrimitiveTaskSpec:
    task_id: str
    primitive_group: str
    dataset_root: str
    dataset_id: str
    default_strength: float = 1.0

    @property
    def family_order(self) -> int:
        return PILOT_PRIMITIVE_FAMILY_ORDER.index(self.primitive_group)


@dataclass(frozen=True)
class ResolvedPrimitiveSelection:
    primitive_groups: tuple[str, ...]
    tasks: tuple[PrimitiveTaskSpec, ...]
    task_strengths: tuple[float, ...]

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks)

    @property
    def is_control_enabled(self) -> bool:
        return any(strength > 0.0 for strength in self.task_strengths)

    @property
    def uses_identity_state(self) -> bool:
        return not self.tasks or not self.is_control_enabled


_PILOT_TASK_SPECS = (
    PrimitiveTaskSpec(
        task_id="detail_sr_x4",
        primitive_group=DEFAULT_PRIMITIVE_GROUP,
        dataset_root="data/detail_sr__RealSR_v3",
        dataset_id="detail_sr__RealSR_v3",
    ),
)

DEFAULT_PRIMITIVE_TASK_ID = _PILOT_TASK_SPECS[0].task_id
_PILOT_TASK_REGISTRY = {task.task_id: task for task in _PILOT_TASK_SPECS}
_PILOT_GROUP_REGISTRY = {task.primitive_group: task for task in _PILOT_TASK_SPECS}
_PILOT_DATASET_REGISTRY = {task.dataset_id: task for task in _PILOT_TASK_SPECS}


def list_catalog_primitive_groups() -> tuple[str, ...]:
    return PILOT_PRIMITIVE_FAMILY_ORDER


def list_supported_primitive_groups() -> tuple[str, ...]:
    return tuple(sorted(_PILOT_GROUP_REGISTRY, key=PILOT_PRIMITIVE_FAMILY_ORDER.index))


def get_task_spec(task_id: str) -> PrimitiveTaskSpec:
    try:
        return _PILOT_TASK_REGISTRY[task_id]
    except KeyError as error:
        supported = ", ".join(sorted(_PILOT_TASK_REGISTRY))
        raise KeyError(f"Unknown ComPhoser pilot task '{task_id}'. Supported task ids: {supported}") from error


def get_task_spec_for_dataset_id(dataset_id: str) -> PrimitiveTaskSpec:
    try:
        return _PILOT_DATASET_REGISTRY[dataset_id]
    except KeyError as error:
        supported = ", ".join(sorted(_PILOT_DATASET_REGISTRY))
        raise KeyError(f"Unknown ComPhoser pilot dataset '{dataset_id}'. Supported dataset ids: {supported}") from error


def resolve_primitive_group(primitive_group: str) -> PrimitiveTaskSpec:
    if primitive_group not in PILOT_PRIMITIVE_FAMILY_ORDER:
        supported = ", ".join(PILOT_PRIMITIVE_FAMILY_ORDER)
        raise ValueError(f"Unknown primitive group '{primitive_group}'. Catalog order: {supported}")

    try:
        return _PILOT_GROUP_REGISTRY[primitive_group]
    except KeyError as error:
        supported = ", ".join(list_supported_primitive_groups())
        raise NotImplementedError(
            f"Primitive group '{primitive_group}' is cataloged but not implemented in Stage 1. "
            f"Stage 1 supports: {supported}"
        ) from error


def normalize_primitive_groups(primitive_groups: Sequence[str] | str | None) -> tuple[str, ...]:
    if primitive_groups is None:
        return ()

    raw_groups = (primitive_groups,) if isinstance(primitive_groups, str) else tuple(primitive_groups)
    seen: set[str] = set()
    validated: list[str] = []

    for primitive_group in raw_groups:
        group = primitive_group.strip()
        if not group:
            raise ValueError("primitive_groups cannot contain empty values")
        if group in seen:
            raise ValueError(f"Duplicate primitive group '{group}' is not allowed")
        resolve_primitive_group(group)
        seen.add(group)
        validated.append(group)

    # Composition order is catalog-owned, not caller-owned.
    return tuple(sorted(validated, key=PILOT_PRIMITIVE_FAMILY_ORDER.index))


def resolve_task_strengths(
    primitive_groups: Sequence[str] | str | None,
    task_strengths: Mapping[str, float] | Sequence[float] | float | None = None,
) -> tuple[float, ...]:
    ordered_groups = normalize_primitive_groups(primitive_groups)
    if not ordered_groups and task_strengths is not None:
        raise ValueError("task_strengths require at least one selected primitive group")

    if not ordered_groups:
        return ()

    if task_strengths is None:
        return tuple(resolve_primitive_group(group).default_strength for group in ordered_groups)

    if isinstance(task_strengths, Mapping):
        valid_keys = set(ordered_groups) | {resolve_primitive_group(group).task_id for group in ordered_groups}
        unexpected = set(task_strengths) - valid_keys
        if unexpected:
            unexpected_keys = ", ".join(sorted(unexpected))
            raise KeyError(f"Unexpected task_strength keys: {unexpected_keys}")

        strengths = []
        for group in ordered_groups:
            task = resolve_primitive_group(group)
            raw_value = task_strengths.get(group, task_strengths.get(task.task_id, task.default_strength))
            strengths.append(_coerce_strength(raw_value, label=group))
        return tuple(strengths)

    if isinstance(task_strengths, (int, float)):
        if len(ordered_groups) != 1:
            raise ValueError("Scalar task_strengths require exactly one selected primitive group")
        return (_coerce_strength(task_strengths, label=ordered_groups[0]),)

    strength_values = tuple(task_strengths)
    if len(strength_values) != len(ordered_groups):
        raise ValueError(
            f"Expected {len(ordered_groups)} task strength values, received {len(strength_values)}"
        )

    return tuple(
        _coerce_strength(strength, label=group) for group, strength in zip(ordered_groups, strength_values)
    )


def resolve_control_selection(
    primitive_groups: Sequence[str] | str | None = None,
    task_strengths: Mapping[str, float] | Sequence[float] | float | None = None,
) -> ResolvedPrimitiveSelection:
    ordered_groups = normalize_primitive_groups(primitive_groups)
    strengths = resolve_task_strengths(ordered_groups, task_strengths)
    tasks = tuple(resolve_primitive_group(group) for group in ordered_groups)

    # v1 composition drops zero-strength groups before downstream routing.
    active_selection = tuple(
        (group, task, strength)
        for group, task, strength in zip(ordered_groups, tasks, strengths)
        if strength > 0.0
    )

    return ResolvedPrimitiveSelection(
        primitive_groups=tuple(group for group, _, _ in active_selection),
        tasks=tuple(task for _, task, _ in active_selection),
        task_strengths=tuple(strength for _, _, strength in active_selection),
    )


def build_pilot_checkpoint_metadata(task_id: str, **overrides: object) -> dict[str, object]:
    task = get_task_spec(task_id)
    metadata: dict[str, object] = {
        "metadata_version": PILOT_CHECKPOINT_METADATA_VERSION,
        "task_id": task.task_id,
        "primitive_group": task.primitive_group,
        "backbone_id": None,
        "query_count": None,
        "query_hidden_size": None,
        "training_dataset_ids": (),
        "prompt_policy_summary": None,
        "evaluation_summary_pointers": (),
        "baseline_comparison_pointers": (),
        "optimization_focus": "interpretability",
    }
    metadata.update(overrides)

    missing = [field for field in PILOT_CHECKPOINT_METADATA_FIELDS if field not in metadata]
    if missing:
        missing_fields = ", ".join(missing)
        raise ValueError(f"Missing checkpoint metadata fields after overrides: {missing_fields}")

    return metadata


def _coerce_strength(value: object, *, label: str) -> float:
    strength = float(value)
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"Task strength for '{label}' must be within [0.0, 1.0], received {strength}")
    return strength


__all__ = [
    "DEFAULT_PRIMITIVE_GROUP",
    "DEFAULT_PRIMITIVE_TASK_ID",
    "PILOT_CHECKPOINT_METADATA_FIELDS",
    "PILOT_CHECKPOINT_METADATA_VERSION",
    "PILOT_PRIMITIVE_FAMILY_ORDER",
    "PrimitiveTaskSpec",
    "ResolvedPrimitiveSelection",
    "build_pilot_checkpoint_metadata",
    "get_task_spec",
    "get_task_spec_for_dataset_id",
    "list_catalog_primitive_groups",
    "list_supported_primitive_groups",
    "normalize_primitive_groups",
    "resolve_control_selection",
    "resolve_primitive_group",
    "resolve_task_strengths",
]

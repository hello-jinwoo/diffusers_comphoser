"""Pilot control registry for ComPhoser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

PILOT_PRIMITIVE_FAMILY_ORDER = ("detail", "tone", "exposure", "depth")
PILOT_QUERIES_PER_PRIMITIVE = 4
PILOT_TOTAL_QUERY_COUNT = len(PILOT_PRIMITIVE_FAMILY_ORDER) * PILOT_QUERIES_PER_PRIMITIVE
PILOT_CHECKPOINT_METADATA_VERSION = "comphoser-fixed-bank-multi-primitive-v1"
DEFAULT_PRIMITIVE_GROUP = "detail"
LEGACY_PRIMITIVE_GROUP_ALIASES = {
    "tone_color": "tone",
}
PILOT_CHECKPOINT_METADATA_FIELDS = (
    "metadata_version",
    "primitive_groups",
    "primitive_family_order",
    "queries_per_primitive",
    "query_count",
    "query_hidden_size",
    "backbone_id",
    "training_task_ids",
    "training_dataset_ids",
    "prompt_policy_summary",
    "evaluation_summary_pointers",
    "baseline_comparison_pointers",
    "optimization_focus",
)


@dataclass(frozen=True)
class PrimitiveFamilySpec:
    primitive_group: str
    default_strength: float = 1.0

    @property
    def family_order(self) -> int:
        return PILOT_PRIMITIVE_FAMILY_ORDER.index(self.primitive_group)

    @property
    def query_slot_start(self) -> int:
        return self.family_order * PILOT_QUERIES_PER_PRIMITIVE

    @property
    def query_slot_stop(self) -> int:
        return self.query_slot_start + PILOT_QUERIES_PER_PRIMITIVE

    @property
    def query_slot_range(self) -> tuple[int, int]:
        return (self.query_slot_start, self.query_slot_stop)

    @property
    def query_slot_indices(self) -> tuple[int, ...]:
        return tuple(range(self.query_slot_start, self.query_slot_stop))


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

    @property
    def is_dataset_ready(self) -> bool:
        return bool(self.dataset_root and self.dataset_id)


@dataclass(frozen=True)
class ResolvedPrimitiveSelection:
    primitive_groups: tuple[str, ...]
    primitive_strengths: tuple[float, ...]
    families: tuple[PrimitiveFamilySpec, ...]
    tasks: tuple[PrimitiveTaskSpec, ...]

    @property
    def task_strengths(self) -> tuple[float, ...]:
        return self.primitive_strengths

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks)

    @property
    def query_slot_ranges(self) -> tuple[tuple[int, int], ...]:
        return tuple(family.query_slot_range for family in self.families)

    @property
    def is_control_enabled(self) -> bool:
        return any(strength > 0.0 for strength in self.primitive_strengths)

    @property
    def uses_identity_state(self) -> bool:
        return not self.primitive_groups or not self.is_control_enabled


_PILOT_FAMILY_SPECS = tuple(PrimitiveFamilySpec(primitive_group=group) for group in PILOT_PRIMITIVE_FAMILY_ORDER)
_PILOT_FAMILY_REGISTRY = {family.primitive_group: family for family in _PILOT_FAMILY_SPECS}

_PILOT_TASK_SPECS = (
    PrimitiveTaskSpec(
        task_id="detail_sr_x4",
        primitive_group=DEFAULT_PRIMITIVE_GROUP,
        dataset_root="data/detail_sr__RealSR_v3",
        dataset_id="detail_sr__RealSR_v3",
    ),
    PrimitiveTaskSpec(
        task_id="tone_style",
        primitive_group="tone",
        dataset_root="data/tone_style__FilmSet",
        dataset_id="tone_style__FilmSet",
    ),
    PrimitiveTaskSpec(
        task_id="exposure_ec",
        primitive_group="exposure",
        dataset_root="data/exposure_ec__MSEC",
        dataset_id="exposure_ec__MSEC",
    ),
    PrimitiveTaskSpec(
        task_id="depth_bokeh",
        primitive_group="depth",
        dataset_root="data/depth_bokeh__RealBokeh",
        dataset_id="depth_bokeh__RealBokeh",
    ),
)

DEFAULT_PRIMITIVE_TASK_ID = _PILOT_TASK_SPECS[0].task_id
_PILOT_TASK_REGISTRY = {task.task_id: task for task in _PILOT_TASK_SPECS}
_PILOT_GROUP_TASK_REGISTRY = {task.primitive_group: task for task in _PILOT_TASK_SPECS}
_PILOT_DATASET_REGISTRY = {task.dataset_id: task for task in _PILOT_TASK_SPECS}


def list_catalog_primitive_groups() -> tuple[str, ...]:
    return PILOT_PRIMITIVE_FAMILY_ORDER


def list_supported_primitive_groups() -> tuple[str, ...]:
    return PILOT_PRIMITIVE_FAMILY_ORDER


def list_dataset_backed_primitive_groups() -> tuple[str, ...]:
    return tuple(sorted(_PILOT_GROUP_TASK_REGISTRY, key=PILOT_PRIMITIVE_FAMILY_ORDER.index))


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


def get_task_spec_for_primitive_group(primitive_group: str) -> PrimitiveTaskSpec | None:
    canonical_group = resolve_primitive_group(primitive_group).primitive_group
    return _PILOT_GROUP_TASK_REGISTRY.get(canonical_group)


def resolve_primitive_group(primitive_group: str) -> PrimitiveFamilySpec:
    canonical_group = _canonicalize_primitive_group(primitive_group)
    try:
        return _PILOT_FAMILY_REGISTRY[canonical_group]
    except KeyError as error:
        supported = ", ".join(PILOT_PRIMITIVE_FAMILY_ORDER)
        raise ValueError(f"Unknown primitive group '{primitive_group}'. Catalog order: {supported}") from error


def normalize_primitive_groups(primitive_groups: Sequence[str] | str | None) -> tuple[str, ...]:
    validated = _validate_primitive_groups_in_input_order(primitive_groups)
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
        normalized_task_strengths = _normalize_task_strength_mapping(task_strengths)
        valid_keys = set(ordered_groups)
        for group in ordered_groups:
            task = get_task_spec_for_primitive_group(group)
            if task is not None:
                valid_keys.add(task.task_id)
        unexpected = set(normalized_task_strengths) - valid_keys
        if unexpected:
            unexpected_keys = ", ".join(sorted(unexpected))
            raise KeyError(f"Unexpected task_strength keys: {unexpected_keys}")

        strengths = []
        for group in ordered_groups:
            family = resolve_primitive_group(group)
            task = get_task_spec_for_primitive_group(group)
            task_id = None if task is None else task.task_id
            raw_value = (
                normalized_task_strengths.get(group, normalized_task_strengths.get(task_id, family.default_strength))
                if task_id is not None
                else normalized_task_strengths.get(group, family.default_strength)
            )
            strengths.append(_coerce_strength(raw_value, label=group))
        return tuple(strengths)

    if isinstance(task_strengths, (int, float)):
        if len(ordered_groups) != 1:
            raise ValueError("Scalar task_strengths require exactly one selected primitive group")
        return (_coerce_strength(task_strengths, label=ordered_groups[0]),)

    input_order_groups = _validate_primitive_groups_in_input_order(primitive_groups)
    strength_values = tuple(task_strengths)
    if len(strength_values) != len(input_order_groups):
        raise ValueError(
            f"Expected {len(input_order_groups)} task strength values, received {len(strength_values)}"
        )

    strength_by_group = {
        group: _coerce_strength(strength, label=group)
        for group, strength in zip(input_order_groups, strength_values)
    }
    return tuple(strength_by_group[group] for group in ordered_groups)


def resolve_control_selection(
    primitive_groups: Sequence[str] | str | None = None,
    task_strengths: Mapping[str, float] | Sequence[float] | float | None = None,
) -> ResolvedPrimitiveSelection:
    ordered_groups = normalize_primitive_groups(primitive_groups)
    strengths = resolve_task_strengths(primitive_groups, task_strengths)
    families = tuple(resolve_primitive_group(group) for group in ordered_groups)

    active_selection = tuple(
        (group, family, strength)
        for group, family, strength in zip(ordered_groups, families, strengths)
        if strength > 0.0
    )
    active_tasks = tuple(
        task
        for group, _, _ in active_selection
        if (task := get_task_spec_for_primitive_group(group)) is not None
    )

    return ResolvedPrimitiveSelection(
        primitive_groups=tuple(group for group, _, _ in active_selection),
        primitive_strengths=tuple(strength for _, _, strength in active_selection),
        families=tuple(family for _, family, _ in active_selection),
        tasks=active_tasks,
    )


def build_pilot_checkpoint_metadata(
    primitive_groups: Sequence[str] | str | None = None,
    *,
    training_task_ids: Sequence[str] = (),
    **overrides: object,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "metadata_version": PILOT_CHECKPOINT_METADATA_VERSION,
        "primitive_groups": normalize_primitive_groups(primitive_groups),
        "primitive_family_order": PILOT_PRIMITIVE_FAMILY_ORDER,
        "queries_per_primitive": PILOT_QUERIES_PER_PRIMITIVE,
        "query_count": PILOT_TOTAL_QUERY_COUNT,
        "query_hidden_size": None,
        "backbone_id": None,
        "training_task_ids": tuple(dict.fromkeys(str(task_id) for task_id in training_task_ids)),
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


def _validate_primitive_groups_in_input_order(
    primitive_groups: Sequence[str] | str | None,
) -> tuple[str, ...]:
    if primitive_groups is None:
        return ()

    raw_groups = (primitive_groups,) if isinstance(primitive_groups, str) else tuple(primitive_groups)
    seen: set[str] = set()
    validated: list[str] = []

    for primitive_group in raw_groups:
        group = _canonicalize_primitive_group(primitive_group)
        if not group:
            raise ValueError("primitive_groups cannot contain empty values")
        if group in seen:
            raise ValueError(f"Duplicate primitive group '{group}' is not allowed")
        resolve_primitive_group(group)
        seen.add(group)
        validated.append(group)

    return tuple(validated)


def _canonicalize_primitive_group(primitive_group: str) -> str:
    normalized_group = primitive_group.strip()
    if not normalized_group:
        return normalized_group
    return LEGACY_PRIMITIVE_GROUP_ALIASES.get(normalized_group, normalized_group)


def _normalize_task_strength_mapping(task_strengths: Mapping[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for raw_key, value in task_strengths.items():
        normalized_key = _normalize_task_strength_key(raw_key)
        existing_value = normalized.get(normalized_key)
        if existing_value is not None and existing_value != value:
            raise KeyError(
                f"Conflicting task_strength values provided for '{raw_key}' via normalized key '{normalized_key}'"
            )
        normalized[normalized_key] = value
    return normalized


def _normalize_task_strength_key(raw_key: str) -> str:
    if raw_key in _PILOT_TASK_REGISTRY:
        return raw_key
    return _canonicalize_primitive_group(raw_key)


__all__ = [
    "DEFAULT_PRIMITIVE_GROUP",
    "DEFAULT_PRIMITIVE_TASK_ID",
    "PILOT_CHECKPOINT_METADATA_FIELDS",
    "PILOT_CHECKPOINT_METADATA_VERSION",
    "PILOT_PRIMITIVE_FAMILY_ORDER",
    "PILOT_QUERIES_PER_PRIMITIVE",
    "PILOT_TOTAL_QUERY_COUNT",
    "PrimitiveFamilySpec",
    "PrimitiveTaskSpec",
    "ResolvedPrimitiveSelection",
    "build_pilot_checkpoint_metadata",
    "get_task_spec",
    "get_task_spec_for_dataset_id",
    "get_task_spec_for_primitive_group",
    "list_catalog_primitive_groups",
    "list_dataset_backed_primitive_groups",
    "list_supported_primitive_groups",
    "normalize_primitive_groups",
    "resolve_control_selection",
    "resolve_primitive_group",
    "resolve_task_strengths",
]

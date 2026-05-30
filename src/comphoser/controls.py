"""Pilot control registry for ComPhoser.

The four-family controller layout (``detail`` / ``tone`` / ``exposure`` / ``depth``)
is hardcoded here — it is the controller contract, not data.

Dataset-backed task specs are **discovered** from the on-disk layout under
``data/<group>_<task_variant>__<dataset_name>/`` rather than declared in code.
Adding a new dataset = create the folder; no code change required.

See ``docs/architecture/training_strategy.md`` for the broader pipeline design
and ``docs/STATUS.md`` for the current registered set.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

PILOT_PRIMITIVE_FAMILY_ORDER = ("detail", "tone", "exposure", "depth")
PILOT_QUERIES_PER_PRIMITIVE = 4
PILOT_TOTAL_QUERY_COUNT = len(PILOT_PRIMITIVE_FAMILY_ORDER) * PILOT_QUERIES_PER_PRIMITIVE
PILOT_CHECKPOINT_METADATA_VERSION = "comphoser-fixed-bank-multi-primitive-v1"
DEFAULT_PRIMITIVE_GROUP = "detail"
DEFAULT_DATA_ROOT = "data"
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
        # Non-catalog specs (empty primitive_group, e.g. downstream_* folders) sort last.
        if self.primitive_group not in PILOT_PRIMITIVE_FAMILY_ORDER:
            return len(PILOT_PRIMITIVE_FAMILY_ORDER)
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


# --------------------------------------------------------------------------- #
# Discovery layer
# --------------------------------------------------------------------------- #


def discover_dataset_task_specs(
    data_root: str | os.PathLike[str] = DEFAULT_DATA_ROOT,
    *,
    include_non_catalog: bool = False,
) -> tuple[PrimitiveTaskSpec, ...]:
    """Walk ``data_root`` and return one ``PrimitiveTaskSpec`` per contract-compliant folder.

    Folder convention: ``<group>_<task_variant>__<dataset_name>/{train,val}/{raw,preprocessed}/...``.

    - ``group`` is the prefix before the first underscore.
    - Both ``train/`` and ``val/`` must exist as directories.
    - ``task_id == dataset_id == <folder name>`` for discovered tasks.
    - Folders starting with ``.`` or ``_`` are ignored.

    ``include_non_catalog`` controls what happens to folders whose ``group`` prefix is
    not in :data:`PILOT_PRIMITIVE_FAMILY_ORDER` (e.g. ``downstream_*``):
    - ``False`` (default) — skipped; the catalog gate filters them out. Catalog-only
      strategies (Stage 1/2, legacy) rely on this so the BCE family target stays well-
      defined.
    - ``True`` — included with ``primitive_group=""``. Used by all-in-one strategy
      (which has no family supervision) so every contract folder enters the task pool.

    Per-folder warnings fire for genuinely malformed folders (bad name, missing splits);
    the by-design non-catalog skip is silent and reported by the caller if it wants.

    The function is pure: it does not consult or update the module-level cache.
    Use :func:`get_dataset_task_specs` for the cached entry point.
    """
    root = Path(data_root)
    if not root.is_dir():
        return ()

    specs: list[PrimitiveTaskSpec] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith(".") or name.startswith("_"):
            continue
        spec = _try_parse_folder(child, data_root, include_non_catalog=include_non_catalog)
        if spec is not None:
            specs.append(spec)
    return tuple(specs)


def _try_parse_folder(
    folder: Path,
    data_root: str | os.PathLike[str],
    *,
    include_non_catalog: bool = False,
) -> PrimitiveTaskSpec | None:
    name = folder.name
    if "__" not in name:
        warnings.warn(
            f"Skipping '{name}': folder name does not contain '__' separator "
            "(contract: <group>_<task_variant>__<dataset_name>)",
            stacklevel=3,
        )
        return None
    prefix, dataset_name = name.split("__", 1)
    if not prefix or not dataset_name:
        warnings.warn(f"Skipping '{name}': empty group/task prefix or dataset suffix", stacklevel=3)
        return None
    group, _, task_variant = prefix.partition("_")
    if not group or not task_variant:
        warnings.warn(
            f"Skipping '{name}': prefix '{prefix}' missing '<group>_<task_variant>' structure",
            stacklevel=3,
        )
        return None
    if group not in PILOT_PRIMITIVE_FAMILY_ORDER:
        if not include_non_catalog:
            # Non-catalog folder, catalog-only discovery requested: skip silently.
            # The catalog gate filter is by-design for Stage 1/2/legacy strategies;
            # all-in-one passes include_non_catalog=True to surface these.
            return None
        if not (folder / "train").is_dir() or not (folder / "val").is_dir():
            warnings.warn(f"Skipping '{name}': missing train/ or val/ split", stacklevel=3)
            return None
        dataset_root_str = f"{os.fspath(data_root).rstrip('/')}/{name}"
        return PrimitiveTaskSpec(
            task_id=name,
            primitive_group="",  # non-catalog → no family; QFormer gets all-zero gate target
            dataset_root=dataset_root_str,
            dataset_id=name,
        )
    if not (folder / "train").is_dir() or not (folder / "val").is_dir():
        warnings.warn(f"Skipping '{name}': missing train/ or val/ split", stacklevel=3)
        return None

    # Preserve the user-supplied data_root format in dataset_root (e.g. "data/foo" not absolute).
    dataset_root_str = f"{os.fspath(data_root).rstrip('/')}/{name}"
    return PrimitiveTaskSpec(
        task_id=name,
        primitive_group=group,
        dataset_root=dataset_root_str,
        dataset_id=name,
    )


_DISCOVERY_CACHE: dict[tuple[str, bool], tuple[PrimitiveTaskSpec, ...]] = {}


def _canonicalize_data_root(data_root: str | os.PathLike[str]) -> str:
    return str(Path(data_root).expanduser().resolve())


def get_dataset_task_specs(
    data_root: str | os.PathLike[str] = DEFAULT_DATA_ROOT,
    *,
    include_non_catalog: bool = False,
) -> tuple[PrimitiveTaskSpec, ...]:
    """Cached discovery output, keyed by (canonicalized ``data_root``, ``include_non_catalog``)."""
    key = (_canonicalize_data_root(data_root), bool(include_non_catalog))
    if key not in _DISCOVERY_CACHE:
        _DISCOVERY_CACHE[key] = discover_dataset_task_specs(
            data_root, include_non_catalog=include_non_catalog
        )
    return _DISCOVERY_CACHE[key]


def reset_dataset_task_cache(data_root: str | os.PathLike[str] | None = None) -> None:
    """Drop cached discovery for ``data_root`` (or all roots when ``None``).

    Use this in tests after creating fixture folders on disk, or to pick up
    new datasets added during a long-lived process.
    """
    if data_root is None:
        _DISCOVERY_CACHE.clear()
        return
    canonical = _canonicalize_data_root(data_root)
    for key in [k for k in _DISCOVERY_CACHE if k[0] == canonical]:
        _DISCOVERY_CACHE.pop(key, None)


def override_dataset_task_specs(
    specs: Sequence[PrimitiveTaskSpec] | None,
    *,
    data_root: str | os.PathLike[str] = DEFAULT_DATA_ROOT,
    include_non_catalog: bool = False,
) -> None:
    """Test/CLI hook: replace cached specs for ``data_root`` (``None`` clears the entry).

    The cache is keyed by ``(data_root, include_non_catalog)``; pass the same
    ``include_non_catalog`` value used by the caller you intend to satisfy. When
    ``specs is None`` all entries for this ``data_root`` are dropped.
    """
    canonical = _canonicalize_data_root(data_root)
    if specs is None:
        for key in [k for k in _DISCOVERY_CACHE if k[0] == canonical]:
            _DISCOVERY_CACHE.pop(key, None)
    else:
        _DISCOVERY_CACHE[(canonical, bool(include_non_catalog))] = tuple(specs)


# --------------------------------------------------------------------------- #
# Lookup helpers (read through the cached discovery)
# --------------------------------------------------------------------------- #


def _registry_by_task_id(
    data_root: str | os.PathLike[str] = DEFAULT_DATA_ROOT,
) -> dict[str, PrimitiveTaskSpec]:
    return {spec.task_id: spec for spec in get_dataset_task_specs(data_root)}


def _registry_by_dataset_id(
    data_root: str | os.PathLike[str] = DEFAULT_DATA_ROOT,
) -> dict[str, PrimitiveTaskSpec]:
    return {spec.dataset_id: spec for spec in get_dataset_task_specs(data_root)}


def list_catalog_primitive_groups() -> tuple[str, ...]:
    """The four hardcoded family slots in their canonical composition order."""
    return PILOT_PRIMITIVE_FAMILY_ORDER


def list_supported_primitive_groups() -> tuple[str, ...]:
    """Alias for the catalog list (kept for callers that distinguish supported vs catalog)."""
    return PILOT_PRIMITIVE_FAMILY_ORDER


def list_dataset_backed_primitive_groups(
    *,
    data_root: str | os.PathLike[str] = DEFAULT_DATA_ROOT,
) -> tuple[str, ...]:
    """Groups with at least one discovered dataset, in canonical family order."""
    groups = {spec.primitive_group for spec in get_dataset_task_specs(data_root)}
    return tuple(group for group in PILOT_PRIMITIVE_FAMILY_ORDER if group in groups)


def get_task_spec(
    task_id: str,
    *,
    data_root: str | os.PathLike[str] = DEFAULT_DATA_ROOT,
) -> PrimitiveTaskSpec:
    registry = _registry_by_task_id(data_root)
    try:
        return registry[task_id]
    except KeyError as error:
        supported = ", ".join(sorted(registry)) or "(none discovered)"
        raise KeyError(
            f"Unknown ComPhoser pilot task '{task_id}'. Discovered task ids: {supported}"
        ) from error


def get_task_spec_for_dataset_id(
    dataset_id: str,
    *,
    data_root: str | os.PathLike[str] = DEFAULT_DATA_ROOT,
) -> PrimitiveTaskSpec:
    registry = _registry_by_dataset_id(data_root)
    try:
        return registry[dataset_id]
    except KeyError as error:
        supported = ", ".join(sorted(registry)) or "(none discovered)"
        raise KeyError(
            f"Unknown ComPhoser pilot dataset '{dataset_id}'. Discovered dataset ids: {supported}"
        ) from error


def resolve_dataset_task_spec_by_id(
    dataset_id: str,
    *,
    data_root: str | os.PathLike[str] = DEFAULT_DATA_ROOT,
) -> PrimitiveTaskSpec:
    """Look up a dataset_id in the discovered catalog or fall back to a synthetic spec
    for a contract-compliant non-catalog folder under ``data_root/``.

    Lets `--train_dataset_ids` / `--validation_dataset_ids` reference any folder that
    follows the on-disk contract (including non-catalog `downstream_*` folders that the
    family-catalog gate would otherwise filter out). The synthetic spec carries an empty
    `primitive_group` — the trainer treats unknown task_ids as no-family, producing an
    all-zero gate_targets mask for those samples.
    """

    try:
        return get_task_spec_for_dataset_id(dataset_id, data_root=data_root)
    except KeyError:
        candidate = Path(os.fspath(data_root)) / dataset_id
        if not candidate.is_dir():
            raise KeyError(
                f"Dataset id '{dataset_id}' is neither a discovered catalog task nor a "
                f"folder under {os.fspath(data_root)}/."
            ) from None
        if not (candidate / "train").is_dir() or not (candidate / "val").is_dir():
            raise KeyError(
                f"Dataset id '{dataset_id}' folder {candidate} is missing train/ or val/ split."
            ) from None
        dataset_root_str = f"{os.fspath(data_root).rstrip('/')}/{dataset_id}"
        return PrimitiveTaskSpec(
            task_id=dataset_id,
            primitive_group="",
            dataset_root=dataset_root_str,
            dataset_id=dataset_id,
        )


def get_task_specs_for_primitive_group(
    primitive_group: str,
    *,
    data_root: str | os.PathLike[str] = DEFAULT_DATA_ROOT,
) -> tuple[PrimitiveTaskSpec, ...]:
    """All discovered tasks in the given group (canonical-family order; empty if none)."""
    canonical_group = resolve_primitive_group(primitive_group).primitive_group
    return tuple(
        spec for spec in get_dataset_task_specs(data_root)
        if spec.primitive_group == canonical_group
    )


def get_default_primitive_task_id(
    *,
    data_root: str | os.PathLike[str] = DEFAULT_DATA_ROOT,
) -> str:
    """First discovered task in canonical family order. Empty string if no datasets."""
    specs = get_dataset_task_specs(data_root)
    if not specs:
        return ""
    ordered = sorted(specs, key=lambda s: (s.family_order, s.task_id))
    return ordered[0].task_id


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
    *,
    data_root: str | os.PathLike[str] = DEFAULT_DATA_ROOT,
) -> tuple[float, ...]:
    ordered_groups = normalize_primitive_groups(primitive_groups)
    if not ordered_groups and task_strengths is not None:
        raise ValueError("task_strengths require at least one selected primitive group")

    if not ordered_groups:
        return ()

    if task_strengths is None:
        return tuple(resolve_primitive_group(group).default_strength for group in ordered_groups)

    if isinstance(task_strengths, Mapping):
        task_id_registry = _registry_by_task_id(data_root)
        normalized_task_strengths = _normalize_task_strength_mapping(task_strengths, task_id_registry)
        valid_keys: set[str] = set(ordered_groups)
        for group in ordered_groups:
            for task in get_task_specs_for_primitive_group(group, data_root=data_root):
                valid_keys.add(task.task_id)
        unexpected = set(normalized_task_strengths) - valid_keys
        if unexpected:
            unexpected_keys = ", ".join(sorted(unexpected))
            raise KeyError(f"Unexpected task_strength keys: {unexpected_keys}")

        strengths: list[float] = []
        for group in ordered_groups:
            family = resolve_primitive_group(group)
            raw_value = normalized_task_strengths.get(group)
            if raw_value is None:
                # Try any task_id within the group (first hit wins; conflicting strengths raise).
                group_task_strengths: dict[str, float] = {}
                for task in get_task_specs_for_primitive_group(group, data_root=data_root):
                    if task.task_id in normalized_task_strengths:
                        group_task_strengths[task.task_id] = normalized_task_strengths[task.task_id]
                if group_task_strengths:
                    values = set(group_task_strengths.values())
                    if len(values) > 1:
                        details = ", ".join(f"{k}={v}" for k, v in sorted(group_task_strengths.items()))
                        raise ValueError(
                            f"Conflicting task_strength values for group '{group}' across its task ids: {details}"
                        )
                    raw_value = next(iter(values))
                else:
                    raw_value = family.default_strength
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
    *,
    data_root: str | os.PathLike[str] = DEFAULT_DATA_ROOT,
) -> ResolvedPrimitiveSelection:
    ordered_groups = normalize_primitive_groups(primitive_groups)
    strengths = resolve_task_strengths(primitive_groups, task_strengths, data_root=data_root)
    families = tuple(resolve_primitive_group(group) for group in ordered_groups)

    active_selection = tuple(
        (group, family, strength)
        for group, family, strength in zip(ordered_groups, families, strengths)
        if strength > 0.0
    )
    active_tasks: list[PrimitiveTaskSpec] = []
    for group, _, _ in active_selection:
        active_tasks.extend(get_task_specs_for_primitive_group(group, data_root=data_root))

    return ResolvedPrimitiveSelection(
        primitive_groups=tuple(group for group, _, _ in active_selection),
        primitive_strengths=tuple(strength for _, _, strength in active_selection),
        families=tuple(family for _, family, _ in active_selection),
        tasks=tuple(active_tasks),
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


def _normalize_task_strength_mapping(
    task_strengths: Mapping[str, float],
    task_id_registry: Mapping[str, PrimitiveTaskSpec],
) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for raw_key, value in task_strengths.items():
        normalized_key = _normalize_task_strength_key(raw_key, task_id_registry)
        existing_value = normalized.get(normalized_key)
        if existing_value is not None and existing_value != value:
            raise KeyError(
                f"Conflicting task_strength values provided for '{raw_key}' via normalized key '{normalized_key}'"
            )
        normalized[normalized_key] = value
    return normalized


def _normalize_task_strength_key(
    raw_key: str,
    task_id_registry: Mapping[str, PrimitiveTaskSpec],
) -> str:
    if raw_key in task_id_registry:
        return raw_key
    return _canonicalize_primitive_group(raw_key)


# Backwards-compatible module-level constant (best-effort at import time).
# Lazy callers should prefer ``get_default_primitive_task_id()``.
try:
    DEFAULT_PRIMITIVE_TASK_ID = get_default_primitive_task_id()
except Exception:  # pragma: no cover - defensive: discovery should not raise
    DEFAULT_PRIMITIVE_TASK_ID = ""


__all__ = [
    "DEFAULT_DATA_ROOT",
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
    "discover_dataset_task_specs",
    "get_dataset_task_specs",
    "get_default_primitive_task_id",
    "get_task_spec",
    "get_task_spec_for_dataset_id",
    "get_task_specs_for_primitive_group",
    "list_catalog_primitive_groups",
    "list_dataset_backed_primitive_groups",
    "list_supported_primitive_groups",
    "normalize_primitive_groups",
    "override_dataset_task_specs",
    "reset_dataset_task_cache",
    "resolve_control_selection",
    "resolve_primitive_group",
    "resolve_task_strengths",
]

"""Stage 1 primitive-training resolution helpers for ComPhoser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .training import TrainingControlSpec, resolve_training_spec


@dataclass(frozen=True)
class PrimitiveTrainingSpec:
    training_spec: TrainingControlSpec
    dataset_roots: tuple[str, ...]

    @property
    def controls(self):
        return self.training_spec.controls

    @property
    def checkpoint_metadata(self) -> tuple[dict[str, object], ...]:
        return self.training_spec.checkpoint_metadata

    @property
    def primary_task_id(self) -> str | None:
        return self.training_spec.task_ids[0] if self.training_spec.task_ids else None


def resolve_primitive_training_spec(
    primitive_groups: Sequence[str] | str | None = None,
    task_strengths: Mapping[str, float] | Sequence[float] | float | None = None,
    metadata_overrides: Mapping[str, Mapping[str, object]] | None = None,
) -> PrimitiveTrainingSpec:
    training_spec = resolve_training_spec(
        primitive_groups=primitive_groups,
        task_strengths=task_strengths,
        metadata_overrides=metadata_overrides,
    )
    dataset_roots = tuple(task.dataset_root for task in training_spec.controls.tasks)
    return PrimitiveTrainingSpec(training_spec=training_spec, dataset_roots=dataset_roots)


__all__ = ["PrimitiveTrainingSpec", "resolve_primitive_training_spec"]

"""Lightweight Q-Former controller for the single-task ComPhoser pilot."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .controls import DEFAULT_PRIMITIVE_TASK_ID

DEFAULT_QFORMER_TASK_ID = DEFAULT_PRIMITIVE_TASK_ID
DEFAULT_QFORMER_QUERY_COUNT = 8
DEFAULT_QFORMER_COND_SUMMARY_TOKENS = 4


@dataclass(frozen=True)
class ComPhoserQFormerOutput:
    query_group: Tensor
    task_strength: Tensor
    raw_query_gates: Tensor
    query_gates: Tensor
    gate_summary: dict[str, Tensor]


@dataclass(frozen=True)
class AugmentedConditioning:
    encoder_hidden_states: Tensor
    txt_ids: Tensor
    added_token_count: int


class ComPhoserQFormer(nn.Module):
    """Single-task pilot controller that gates learned queries from prompt and image context."""

    def __init__(
        self,
        hidden_size: int,
        *,
        cond_token_dim: int | None = None,
        num_queries: int = DEFAULT_QFORMER_QUERY_COUNT,
        cond_summary_tokens: int = DEFAULT_QFORMER_COND_SUMMARY_TOKENS,
        num_heads: int = 4,
        ffn_multiplier: int = 4,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_queries <= 0:
            raise ValueError("num_queries must be positive")
        if cond_summary_tokens <= 0:
            raise ValueError("cond_summary_tokens must be positive")
        if num_heads <= 0 or hidden_size % num_heads != 0:
            raise ValueError("num_heads must divide hidden_size")

        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.cond_summary_tokens = cond_summary_tokens
        self.cond_token_dim = hidden_size if cond_token_dim is None else cond_token_dim
        self.num_heads = num_heads
        self.ffn_multiplier = ffn_multiplier

        if self.cond_token_dim == hidden_size:
            self.cond_projection: nn.Module = nn.Identity()
        else:
            self.cond_projection = nn.Linear(self.cond_token_dim, hidden_size)

        self.prompt_norm = nn.LayerNorm(hidden_size)
        self.cond_norm = nn.LayerNorm(hidden_size)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.task_embedding = nn.Embedding(1, hidden_size)
        self.query_bank = nn.Parameter(torch.empty(num_queries, hidden_size))
        self.shared_trunk = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=max(hidden_size * ffn_multiplier, hidden_size),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.query_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_bank, mean=0.0, std=0.02)
        nn.init.normal_(self.task_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        prompt_embeds: Tensor,
        cond_tokens: Tensor,
        *,
        task_ids: str | tuple[str, ...] | list[str] | None = None,
        task_strengths: Tensor | float | tuple[float, ...] | list[float] | None = None,
    ) -> ComPhoserQFormerOutput:
        if prompt_embeds.ndim != 3:
            raise ValueError("prompt_embeds must have shape [batch, seq_len, hidden_size]")
        if cond_tokens.ndim != 3:
            raise ValueError("cond_tokens must have shape [batch, seq_len, cond_hidden_size]")
        if prompt_embeds.shape[0] != cond_tokens.shape[0]:
            raise ValueError("prompt_embeds and cond_tokens must share the same batch size")
        if prompt_embeds.shape[-1] != self.hidden_size:
            raise ValueError(
                f"prompt_embeds hidden size must match controller hidden_size={self.hidden_size}, "
                f"received {prompt_embeds.shape[-1]}"
            )
        if cond_tokens.shape[-1] != self.cond_token_dim:
            raise ValueError(
                f"cond_tokens hidden size must match controller cond_token_dim={self.cond_token_dim}, "
                f"received {cond_tokens.shape[-1]}"
            )

        batch_size = prompt_embeds.shape[0]
        normalized_task_ids = _normalize_task_ids(task_ids, batch_size=batch_size)
        if normalized_task_ids and any(task_id != DEFAULT_QFORMER_TASK_ID for task_id in normalized_task_ids):
            unsupported = ", ".join(sorted(set(normalized_task_ids)))
            raise NotImplementedError(
                f"ComPhoserQFormer only supports '{DEFAULT_QFORMER_TASK_ID}', received: {unsupported}"
            )

        task_strength = _coerce_task_strengths(
            task_strengths,
            batch_size=batch_size,
            device=prompt_embeds.device,
            dtype=prompt_embeds.dtype,
            default_value=0.0 if not normalized_task_ids else 1.0,
        )

        prompt_context = self.prompt_norm(prompt_embeds)
        cond_context = self.cond_norm(self._summarize_condition_tokens(cond_tokens))
        task_context = self.task_embedding(
            torch.zeros(batch_size, dtype=torch.long, device=prompt_embeds.device)
        ).unsqueeze(1)
        joint_context = torch.cat((prompt_context, cond_context, task_context), dim=1)
        joint_context = self.shared_trunk(joint_context)

        query_bank = self.query_bank.unsqueeze(0).expand(batch_size, -1, -1)
        attended_context, _ = self.query_attention(query_bank, joint_context, joint_context, need_weights=False)
        raw_query_gates = self.gate_head(attended_context + query_bank).squeeze(-1)

        base_query_gates = torch.sigmoid(raw_query_gates)
        query_gates = base_query_gates * task_strength.unsqueeze(-1)
        query_group = self.query_norm(query_bank) * query_gates.unsqueeze(-1)

        gate_summary = {
            "raw_mean": raw_query_gates.mean(dim=1),
            "raw_std": raw_query_gates.std(dim=1, unbiased=False),
            "active_mean": query_gates.mean(dim=1),
            "active_min": query_gates.min(dim=1).values,
            "active_max": query_gates.max(dim=1).values,
        }
        return ComPhoserQFormerOutput(
            query_group=query_group,
            task_strength=task_strength,
            raw_query_gates=raw_query_gates,
            query_gates=query_gates,
            gate_summary=gate_summary,
        )

    def _summarize_condition_tokens(self, cond_tokens: Tensor) -> Tensor:
        cond_hidden_states = self.cond_projection(cond_tokens)
        if cond_hidden_states.shape[1] == 0:
            return cond_hidden_states.new_zeros(cond_hidden_states.shape[0], 1, self.hidden_size)
        if cond_hidden_states.shape[1] <= self.cond_summary_tokens:
            return cond_hidden_states

        return F.adaptive_avg_pool1d(
            cond_hidden_states.transpose(1, 2),
            output_size=self.cond_summary_tokens,
        ).transpose(1, 2)


def build_synthetic_txt_ids(base_txt_ids: Tensor, added_token_count: int) -> Tensor:
    if added_token_count < 0:
        raise ValueError("added_token_count cannot be negative")
    if base_txt_ids.ndim not in (2, 3):
        raise ValueError("base_txt_ids must have shape [seq_len, 4] or [batch, seq_len, 4]")
    if base_txt_ids.shape[-1] != 4:
        raise ValueError("base_txt_ids must use the FLUX [*, *, 4] position-id contract")

    if base_txt_ids.ndim == 2:
        return _build_synthetic_txt_ids_2d(base_txt_ids, added_token_count)
    return _build_synthetic_txt_ids_3d(base_txt_ids, added_token_count)


def append_query_tokens_to_prompt(
    prompt_embeds: Tensor,
    txt_ids: Tensor,
    query_group: Tensor | None,
) -> AugmentedConditioning:
    if prompt_embeds.ndim != 3:
        raise ValueError("prompt_embeds must have shape [batch, seq_len, hidden_size]")
    if txt_ids.ndim not in (2, 3):
        raise ValueError("txt_ids must have shape [seq_len, 4] or [batch, seq_len, 4]")
    if query_group is None:
        return AugmentedConditioning(
            encoder_hidden_states=prompt_embeds,
            txt_ids=txt_ids,
            added_token_count=0,
        )
    if query_group.ndim != 3:
        raise ValueError("query_group must have shape [batch, num_queries, hidden_size]")
    if query_group.shape[0] != prompt_embeds.shape[0]:
        raise ValueError("query_group and prompt_embeds must share the same batch size")
    if query_group.shape[-1] != prompt_embeds.shape[-1]:
        raise ValueError("query_group hidden size must match prompt_embeds hidden size")
    if query_group.shape[1] == 0:
        return AugmentedConditioning(
            encoder_hidden_states=prompt_embeds,
            txt_ids=txt_ids,
            added_token_count=0,
        )
    if txt_ids.ndim == 3 and txt_ids.shape[0] != prompt_embeds.shape[0]:
        raise ValueError("3D txt_ids must share the same batch size as prompt_embeds")

    synthetic_txt_ids = build_synthetic_txt_ids(txt_ids, query_group.shape[1])
    cat_dim = 1 if txt_ids.ndim == 3 else 0
    return AugmentedConditioning(
        encoder_hidden_states=torch.cat((prompt_embeds, query_group.to(dtype=prompt_embeds.dtype)), dim=1),
        txt_ids=torch.cat((txt_ids, synthetic_txt_ids), dim=cat_dim),
        added_token_count=query_group.shape[1],
    )


def _build_synthetic_txt_ids_2d(base_txt_ids: Tensor, added_token_count: int) -> Tensor:
    if added_token_count == 0:
        return base_txt_ids.new_empty((0, 4))

    synthetic_txt_ids = base_txt_ids.new_zeros((added_token_count, 4))
    start_index = int(base_txt_ids[:, 3].max().item()) + 1 if base_txt_ids.shape[0] else 0
    synthetic_txt_ids[:, 3] = torch.arange(
        start_index,
        start_index + added_token_count,
        device=base_txt_ids.device,
        dtype=base_txt_ids.dtype,
    )
    return synthetic_txt_ids


def _build_synthetic_txt_ids_3d(base_txt_ids: Tensor, added_token_count: int) -> Tensor:
    batch_size = base_txt_ids.shape[0]
    if added_token_count == 0:
        return base_txt_ids.new_empty((batch_size, 0, 4))

    synthetic_txt_ids = base_txt_ids.new_zeros((batch_size, added_token_count, 4))
    if base_txt_ids.shape[1] == 0:
        start_index = torch.zeros(batch_size, device=base_txt_ids.device, dtype=base_txt_ids.dtype)
    else:
        start_index = base_txt_ids[:, :, 3].amax(dim=1) + 1

    token_offsets = torch.arange(
        added_token_count,
        device=base_txt_ids.device,
        dtype=base_txt_ids.dtype,
    ).unsqueeze(0)
    synthetic_txt_ids[:, :, 3] = start_index.unsqueeze(1) + token_offsets
    return synthetic_txt_ids


def _normalize_task_ids(
    task_ids: str | tuple[str, ...] | list[str] | None,
    *,
    batch_size: int,
) -> tuple[str, ...]:
    if task_ids is None:
        return (DEFAULT_QFORMER_TASK_ID,) * batch_size
    if isinstance(task_ids, str):
        return (task_ids,) * batch_size

    normalized_task_ids = tuple(task_ids)
    if not normalized_task_ids:
        return ()
    if len(normalized_task_ids) == 1:
        return normalized_task_ids * batch_size
    if len(normalized_task_ids) != batch_size:
        raise ValueError(
            f"Expected task_ids to have length 1 or batch size {batch_size}, received {len(normalized_task_ids)}"
        )
    return normalized_task_ids


def _coerce_task_strengths(
    task_strengths: Tensor | float | tuple[float, ...] | list[float] | None,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    default_value: float,
) -> Tensor:
    if task_strengths is None:
        return torch.full((batch_size,), default_value, device=device, dtype=dtype)

    if isinstance(task_strengths, Tensor):
        strengths = task_strengths.to(device=device, dtype=dtype)
        if strengths.ndim == 0:
            strengths = strengths.expand(batch_size)
        elif strengths.ndim == 1 and strengths.shape[0] == 1:
            strengths = strengths.expand(batch_size)
        elif strengths.ndim == 1 and strengths.shape[0] == 0:
            strengths = torch.zeros(batch_size, device=device, dtype=dtype)
        elif strengths.ndim != 1 or strengths.shape[0] != batch_size:
            raise ValueError(
                f"Expected task_strengths tensor to have shape [] or [{batch_size}], received {tuple(strengths.shape)}"
            )
    elif isinstance(task_strengths, (int, float)):
        strengths = torch.full((batch_size,), float(task_strengths), device=device, dtype=dtype)
    else:
        strengths = torch.tensor(tuple(float(value) for value in task_strengths), device=device, dtype=dtype)
        if strengths.shape[0] == 0:
            strengths = torch.zeros(batch_size, device=device, dtype=dtype)
        elif strengths.shape[0] == 1:
            strengths = strengths.expand(batch_size)
        elif strengths.shape[0] != batch_size:
            raise ValueError(
                f"Expected task_strengths to have length 1 or batch size {batch_size}, received {strengths.shape[0]}"
            )

    if torch.any(strengths < 0.0) or torch.any(strengths > 1.0):
        raise ValueError("task_strengths must stay within [0.0, 1.0]")
    return strengths


__all__ = [
    "AugmentedConditioning",
    "ComPhoserQFormer",
    "ComPhoserQFormerOutput",
    "DEFAULT_QFORMER_COND_SUMMARY_TOKENS",
    "DEFAULT_QFORMER_QUERY_COUNT",
    "DEFAULT_QFORMER_TASK_ID",
    "append_query_tokens_to_prompt",
    "build_synthetic_txt_ids",
]

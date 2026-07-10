"""Fixed-bank prompt-routed controller for the v1 ComPhoser multi-primitive path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from .controls import (
    DEFAULT_PRIMITIVE_TASK_ID,
    PILOT_PRIMITIVE_FAMILY_ORDER,
    PILOT_QUERIES_PER_PRIMITIVE,
    PILOT_TOTAL_QUERY_COUNT,
    ResolvedPrimitiveSelection,
    resolve_control_selection,
    resolve_primitive_group,
)


DEFAULT_QFORMER_TASK_ID = DEFAULT_PRIMITIVE_TASK_ID
DEFAULT_QFORMER_QUERY_COUNT = PILOT_TOTAL_QUERY_COUNT
DEFAULT_QFORMER_COND_SUMMARY_TOKENS = 4
DEFAULT_QFORMER_NUM_LAYERS = 3
QFORMER_CONTROLLER_LAYOUT_PROMPT_ROUTER_V2 = "fixed_global_query_bank_prompt_router_v2"
QFORMER_CONTROLLER_LAYOUT_PROMPT_IMAGE_ROUTER_V3 = "fixed_global_query_bank_prompt_image_router_v3"


def _resolve_cond_pool_heads(cond_token_dim: int) -> int:
    """Largest head count in {4, 2, 1} that divides ``cond_token_dim`` (128 -> 4)."""

    for candidate in (4, 2, 1):
        if cond_token_dim % candidate == 0:
            return candidate
    return 1


@dataclass(frozen=True)
class ComPhoserQFormerOutput:
    query_group: Tensor
    gate_targets: Tensor
    raw_query_gates: Tensor
    predicted_query_gates: Tensor
    query_gates: Tensor
    gate_summary: dict[str, Tensor]


@dataclass(frozen=True)
class AugmentedConditioning:
    encoder_hidden_states: Tensor
    txt_ids: Tensor
    added_token_count: int


class ComPhoserQFormer(nn.Module):
    """Prompt-only QWP-Net over one fixed global query bank."""

    def __init__(
        self,
        hidden_size: int,
        *,
        cond_token_dim: int | None = None,
        num_queries: int = DEFAULT_QFORMER_QUERY_COUNT,
        queries_per_primitive: int = PILOT_QUERIES_PER_PRIMITIVE,
        cond_summary_tokens: int = DEFAULT_QFORMER_COND_SUMMARY_TOKENS,
        num_layers: int = DEFAULT_QFORMER_NUM_LAYERS,
        num_heads: int = 16,
        ffn_multiplier: int = 4,
        routing_dim: int | None = None,
        gate_head_hidden: int | None = None,
        output_content_mix: bool = False,
        routing_rounds: int = 1,
        routing_mean_pool: bool = False,
        image_routing: bool = False,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        resolved_qpf = int(queries_per_primitive)
        if resolved_qpf <= 0:
            raise ValueError("queries_per_primitive must be positive")
        derived_num_queries = len(PILOT_PRIMITIVE_FAMILY_ORDER) * resolved_qpf
        if num_queries != derived_num_queries:
            raise ValueError(
                f"num_queries ({num_queries}) must equal {len(PILOT_PRIMITIVE_FAMILY_ORDER)} families "
                f"x queries_per_primitive ({resolved_qpf}) = {derived_num_queries}"
            )
        if cond_summary_tokens <= 0:
            raise ValueError("cond_summary_tokens must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        # routing_dim (optional bottleneck): the routing/gate-prediction path (trunk + query
        # attention + gate head) runs at routing_dim instead of hidden_size, while the output query
        # bank stays at hidden_size (so the appended conditioning tokens are unchanged). The trunk
        # dominates the parameter count (~hidden_size^2 per layer), so a small routing_dim shrinks
        # the controller ~ (routing_dim/hidden_size)^2. routing_dim=None (or ==hidden_size) keeps the
        # original full-width path, byte-for-byte, so legacy checkpoints stay key-compatible.
        resolved_routing_dim = hidden_size if routing_dim in (None, hidden_size) else int(routing_dim)
        if resolved_routing_dim <= 0:
            raise ValueError("routing_dim must be positive")
        if num_heads <= 0 or resolved_routing_dim % num_heads != 0:
            raise ValueError("num_heads must divide routing_dim (== hidden_size when no bottleneck)")

        self.hidden_size = hidden_size
        self.routing_dim = resolved_routing_dim
        self._bottleneck = resolved_routing_dim != hidden_size
        self.num_queries = num_queries
        self.queries_per_primitive = resolved_qpf
        self.cond_summary_tokens = cond_summary_tokens
        self.cond_token_dim = hidden_size if cond_token_dim is None else cond_token_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_multiplier = ffn_multiplier
        # Routing mechanism (no extra params — all share the single query_attention):
        #   routing_rounds>1 => iterative refinement (queries re-attend the prompt N times);
        #   routing_mean_pool => skip query cross-attention, use the mean-pooled prompt context.
        self.routing_rounds = int(routing_rounds)
        if self.routing_rounds < 1:
            raise ValueError("routing_rounds must be >= 1")
        self.routing_mean_pool = bool(routing_mean_pool)

        self.prompt_norm = nn.LayerNorm(hidden_size)
        self.query_norm = nn.LayerNorm(hidden_size)  # normalizes the output bank (hidden_size)
        self.query_bank = nn.Parameter(torch.empty(num_queries, hidden_size))
        if self._bottleneck:
            self.prompt_proj = nn.Linear(hidden_size, self.routing_dim)
            self.query_proj = nn.Linear(hidden_size, self.routing_dim)  # bank -> routing space
            self.query_router_norm = nn.LayerNorm(self.routing_dim)
        self.shared_trunk = self._build_trunk_layer()
        self.extra_trunk_layers = nn.ModuleList(self._build_trunk_layer() for _ in range(num_layers - 1))
        self.query_attention = nn.MultiheadAttention(
            embed_dim=self.routing_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        # Gate head: default is LayerNorm -> Linear(->1). gate_head_hidden inserts a hidden layer
        # (LayerNorm -> Linear -> GELU -> Linear) for more gate-prediction capacity.
        self.gate_head_hidden = None if not gate_head_hidden else int(gate_head_hidden)
        if self.gate_head_hidden:
            self.gate_head = nn.Sequential(
                nn.LayerNorm(self.routing_dim),
                nn.Linear(self.routing_dim, self.gate_head_hidden),
                nn.GELU(),
                nn.Linear(self.gate_head_hidden, 1),
            )
        else:
            self.gate_head = nn.Sequential(
                nn.LayerNorm(self.routing_dim),
                nn.Linear(self.routing_dim, 1),
            )

        # Output design: by default the appended tokens are the gated static bank. output_content_mix
        # additively blends the prompt-attended (routing) context into the output bank so the injected
        # tokens become prompt-adaptive content, not just a gated static bank. content_scale starts at
        # 0 so the module is a no-op at init (baseline-equivalent), then learns how much content to mix.
        self.output_content_mix = bool(output_content_mix)
        if self.output_content_mix:
            self.content_proj = nn.Linear(self.routing_dim, hidden_size)
            self.content_scale = nn.Parameter(torch.zeros(1))

        # Optional condition-image-aware routing. The image latent (``cond_tokens``, a long
        # [batch, H*W, cond_token_dim] sequence) is pooled to ``cond_summary_tokens`` tokens by a
        # small learnable attention-pool in the cheap cond_token_dim space, then projected to
        # hidden_size and concatenated onto the prompt context that the query bank attends over for
        # gate prediction. The gated-bank output path is unchanged, so this only widens the routing
        # signal. Modules are created only when enabled so prompt-only checkpoints stay key-compatible.
        self.image_routing = bool(image_routing)
        if self.image_routing and self._bottleneck:
            raise NotImplementedError("routing_dim bottleneck + image_routing are not supported together yet")
        if self.image_routing:
            pool_heads = _resolve_cond_pool_heads(self.cond_token_dim)
            self.cond_pool_num_heads = pool_heads
            self.cond_pool_queries = nn.Parameter(torch.empty(self.cond_summary_tokens, self.cond_token_dim))
            self.cond_pool_norm = nn.LayerNorm(self.cond_token_dim)
            self.cond_pool_attention = nn.MultiheadAttention(
                embed_dim=self.cond_token_dim,
                num_heads=pool_heads,
                dropout=0.0,
                batch_first=True,
            )
            self.cond_proj = nn.Linear(self.cond_token_dim, hidden_size)
            self.cond_summary_norm = nn.LayerNorm(hidden_size)

        self.reset_parameters()

    def _build_trunk_layer(self) -> nn.TransformerEncoderLayer:
        return nn.TransformerEncoderLayer(
            d_model=self.routing_dim,
            nhead=self.num_heads,
            dim_feedforward=max(self.routing_dim * self.ffn_multiplier, self.routing_dim),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_bank, mean=0.0, std=0.02)
        if self.image_routing:
            nn.init.normal_(self.cond_pool_queries, mean=0.0, std=0.02)

    def forward(
        self,
        prompt_embeds: Tensor,
        cond_tokens: Tensor | None = None,
        *,
        primitive_groups: (Sequence[Sequence[str] | str] | Sequence[str] | str | None) = None,
        primitive_strengths: (
            Sequence[Sequence[float] | float] | Sequence[float] | Mapping[str, float] | Tensor | float | None
        ) = None,
        explicit_token_masking: Sequence[float] | Tensor | None = None,
    ) -> ComPhoserQFormerOutput:
        if prompt_embeds.ndim != 3:
            raise ValueError("prompt_embeds must have shape [batch, seq_len, hidden_size]")
        if prompt_embeds.shape[-1] != self.hidden_size:
            raise ValueError(
                f"prompt_embeds hidden size must match controller hidden_size={self.hidden_size}, "
                f"received {prompt_embeds.shape[-1]}"
            )

        batch_size = prompt_embeds.shape[0]
        if cond_tokens is not None:
            if cond_tokens.ndim != 3:
                raise ValueError("cond_tokens must have shape [batch, seq_len, cond_hidden_size]")
            if cond_tokens.shape[0] != batch_size:
                raise ValueError("prompt_embeds and cond_tokens must share the same batch size")
            if cond_tokens.shape[-1] != self.cond_token_dim:
                raise ValueError(
                    f"cond_tokens hidden size must match controller cond_token_dim={self.cond_token_dim}, "
                    f"received {cond_tokens.shape[-1]}"
                )

        gate_targets = build_batch_query_gate_target_mask(
            primitive_groups,
            primitive_strengths,
            batch_size=batch_size,
            queries_per_primitive=self.queries_per_primitive,
            device=prompt_embeds.device,
            dtype=prompt_embeds.dtype,
        )

        prompt_context = self.prompt_norm(prompt_embeds)
        if self._bottleneck:
            prompt_context = self.prompt_proj(prompt_context)
        prompt_context = self.shared_trunk(prompt_context)
        for layer in self.extra_trunk_layers:
            prompt_context = layer(prompt_context)
        routing_context = self._build_routing_context(prompt_context, cond_tokens)
        query_bank = self.query_bank.unsqueeze(0).expand(batch_size, -1, -1)
        normalized_query_bank = self.query_norm(query_bank)  # output-space bank (hidden_size)
        # Routing queries live in routing_dim: the full-width path attends the bank directly; the
        # bottleneck path projects the bank into routing_dim first. The gate head residual uses the
        # same normalized routing-space representation that is attended (A1 / R08).
        if self._bottleneck:
            router_query = self.query_proj(query_bank)
            router_residual = self.query_router_norm(router_query)
        else:
            router_query = query_bank
            router_residual = normalized_query_bank
        if self.routing_mean_pool:
            # no per-query cross-attention: broadcast the mean-pooled prompt context to every slot
            attended_context = routing_context.mean(dim=1, keepdim=True).expand(-1, router_query.shape[1], -1)
        else:
            attended_context, _ = self.query_attention(
                router_query, routing_context, routing_context, need_weights=False
            )
            for _ in range(self.routing_rounds - 1):  # iterative refinement (weight-shared)
                attended_context, _ = self.query_attention(
                    attended_context, routing_context, routing_context, need_weights=False
                )
        raw_query_gates = self.gate_head(attended_context + router_residual).squeeze(-1)

        predicted_query_gates = torch.sigmoid(raw_query_gates)
        normalized_override = _normalize_explicit_token_masking(
            explicit_token_masking,
            batch_size=batch_size,
            num_queries=self.num_queries,
            device=predicted_query_gates.device,
            dtype=predicted_query_gates.dtype,
        )
        query_gates = predicted_query_gates if normalized_override is None else normalized_override
        if self.output_content_mix:
            # prompt-adaptive tokens: blend attended routing context into the output bank
            output_bank = self.query_norm(query_bank + self.content_scale * self.content_proj(attended_context))
        else:
            output_bank = normalized_query_bank
        query_group = output_bank * query_gates.unsqueeze(-1)

        return ComPhoserQFormerOutput(
            query_group=query_group,
            gate_targets=gate_targets,
            raw_query_gates=raw_query_gates,
            predicted_query_gates=predicted_query_gates,
            query_gates=query_gates,
            gate_summary=_build_gate_summary(
                raw_query_gates,
                predicted_query_gates,
                query_gates,
                gate_targets,
                queries_per_primitive=self.queries_per_primitive,
                explicit_token_masking=normalized_override,
            ),
        )

    def _build_routing_context(self, prompt_context: Tensor, cond_tokens: Tensor | None) -> Tensor:
        """Return the cross-attention context the query bank routes over.

        Prompt-only by default. When ``image_routing`` is enabled and ``cond_tokens`` are supplied,
        the image latent is attention-pooled to a few summary tokens and concatenated onto the
        prompt context; when it is enabled but ``cond_tokens`` is ``None`` it falls back to
        prompt-only so prompt-only evaluation still works.
        """

        if not self.image_routing or cond_tokens is None:
            return prompt_context

        batch_size = prompt_context.shape[0]
        kv = self.cond_pool_norm(cond_tokens.to(dtype=prompt_context.dtype))
        pool_queries = self.cond_pool_queries.to(dtype=prompt_context.dtype)
        pool_queries = pool_queries.unsqueeze(0).expand(batch_size, -1, -1)
        cond_summary, _ = self.cond_pool_attention(pool_queries, kv, kv, need_weights=False)
        cond_context = self.cond_summary_norm(self.cond_proj(cond_summary))
        return torch.cat([prompt_context, cond_context], dim=1)


def build_query_gate_target_mask(
    primitive_groups: Sequence[str] | str | None = None,
    primitive_strengths: Mapping[str, float] | Sequence[float] | float | None = None,
    *,
    queries_per_primitive: int = PILOT_QUERIES_PER_PRIMITIVE,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    selection = resolve_control_selection(primitive_groups=primitive_groups, task_strengths=primitive_strengths)
    return _selection_to_gate_mask(selection, queries_per_primitive=queries_per_primitive, device=device, dtype=dtype)


def build_batch_query_gate_target_mask(
    primitive_groups: Sequence[Sequence[str] | str] | Sequence[str] | str | None,
    primitive_strengths: Sequence[Sequence[float] | float]
    | Sequence[float]
    | Mapping[str, float]
    | Tensor
    | float
    | None,
    *,
    batch_size: int,
    queries_per_primitive: int = PILOT_QUERIES_PER_PRIMITIVE,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    resolved_dtype = dtype or torch.float32
    selections = _resolve_batch_control_selections(
        primitive_groups,
        primitive_strengths,
        batch_size=batch_size,
    )
    return torch.stack(
        [
            _selection_to_gate_mask(
                selection, queries_per_primitive=queries_per_primitive, device=device, dtype=resolved_dtype
            )
            for selection in selections
        ],
        dim=0,
    )


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


def _selection_to_gate_mask(
    selection: ResolvedPrimitiveSelection,
    *,
    queries_per_primitive: int = PILOT_QUERIES_PER_PRIMITIVE,
    device: torch.device | None,
    dtype: torch.dtype | None,
) -> Tensor:
    total = len(PILOT_PRIMITIVE_FAMILY_ORDER) * queries_per_primitive
    mask = torch.zeros(total, device=device, dtype=dtype or torch.float32)
    for primitive_group, strength in zip(selection.primitive_groups, selection.primitive_strengths):
        family = resolve_primitive_group(primitive_group)
        start = family.family_order * queries_per_primitive
        mask[start : start + queries_per_primitive] = float(strength)
    return mask


def _resolve_batch_control_selections(
    primitive_groups: Sequence[Sequence[str] | str] | Sequence[str] | str | None,
    primitive_strengths: Sequence[Sequence[float] | float]
    | Sequence[float]
    | Mapping[str, float]
    | Tensor
    | float
    | None,
    *,
    batch_size: int,
) -> tuple[ResolvedPrimitiveSelection, ...]:
    if primitive_groups is None or isinstance(primitive_groups, str) or _is_flat_string_sequence(primitive_groups):
        selection = resolve_control_selection(
            primitive_groups=primitive_groups,
            task_strengths=_normalize_empty_strength_spec(primitive_strengths),
        )
        return (selection,) * batch_size

    normalized_group_batches = tuple(primitive_groups)
    if len(normalized_group_batches) != batch_size:
        raise ValueError(
            f"Expected batched primitive_groups to have length {batch_size}, received {len(normalized_group_batches)}"
        )

    if primitive_strengths is None:
        strength_batches: tuple[object, ...] = (None,) * batch_size
    else:
        if isinstance(primitive_strengths, Tensor):
            if primitive_strengths.ndim == 0:
                strength_batches = (float(primitive_strengths.item()),) * batch_size
            elif primitive_strengths.ndim == 1 and primitive_strengths.shape[0] == batch_size:
                strength_batches = tuple(float(value) for value in primitive_strengths.tolist())
            else:
                raise ValueError(
                    "Tensor primitive_strengths must be scalar or have shape [batch_size] when using batched groups"
                )
        else:
            strength_batches = tuple(primitive_strengths)
            if len(strength_batches) != batch_size:
                raise ValueError(
                    f"Expected batched primitive_strengths to have length {batch_size}, "
                    f"received {len(strength_batches)}"
                )

    selections: list[ResolvedPrimitiveSelection] = []
    for sample_groups, sample_strengths in zip(normalized_group_batches, strength_batches):
        selections.append(
            resolve_control_selection(
                primitive_groups=sample_groups,
                task_strengths=_normalize_empty_strength_spec(sample_strengths),
            )
        )
    return tuple(selections)


def _build_gate_summary(
    raw_query_gates: Tensor,
    predicted_query_gates: Tensor,
    query_gates: Tensor,
    gate_targets: Tensor,
    *,
    queries_per_primitive: int = PILOT_QUERIES_PER_PRIMITIVE,
    explicit_token_masking: Tensor | None,
) -> dict[str, Tensor]:
    family_shape = (-1, len(PILOT_PRIMITIVE_FAMILY_ORDER), queries_per_primitive)
    raw_family = raw_query_gates.reshape(family_shape)
    predicted_family = predicted_query_gates.reshape(family_shape)
    active_family = query_gates.reshape(family_shape)
    target_family = gate_targets.reshape(family_shape)
    summary = {
        "raw_mean": raw_query_gates.mean(dim=1),
        "raw_std": raw_query_gates.std(dim=1, unbiased=False),
        "predicted_mean": predicted_query_gates.mean(dim=1),
        "predicted_min": predicted_query_gates.min(dim=1).values,
        "predicted_max": predicted_query_gates.max(dim=1).values,
        "active_mean": query_gates.mean(dim=1),
        "active_min": query_gates.min(dim=1).values,
        "active_max": query_gates.max(dim=1).values,
        "effective_mean": query_gates.mean(dim=1),
        "effective_min": query_gates.min(dim=1).values,
        "effective_max": query_gates.max(dim=1).values,
        "target_mean": gate_targets.mean(dim=1),
        "family_raw_mean": raw_family.mean(dim=2),
        "family_predicted_mean": predicted_family.mean(dim=2),
        "family_active_mean": active_family.mean(dim=2),
        "family_effective_mean": active_family.mean(dim=2),
        "family_target_mean": target_family.mean(dim=2),
        "explicit_token_masking_applied": torch.full(
            (raw_query_gates.shape[0],),
            explicit_token_masking is not None,
            device=raw_query_gates.device,
            dtype=torch.bool,
        ),
    }
    if explicit_token_masking is not None:
        summary["explicit_token_masking"] = explicit_token_masking
    return summary


def _normalize_explicit_token_masking(
    explicit_token_masking: Sequence[float] | Tensor | None,
    *,
    batch_size: int,
    num_queries: int = PILOT_TOTAL_QUERY_COUNT,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor | None:
    if explicit_token_masking is None:
        return None

    if isinstance(explicit_token_masking, Tensor):
        mask = explicit_token_masking.to(device=device, dtype=dtype)
    else:
        mask = torch.as_tensor(tuple(float(value) for value in explicit_token_masking), device=device, dtype=dtype)

    if mask.ndim == 1:
        if mask.shape[0] != num_queries:
            raise ValueError(
                f"explicit_token_masking must contain exactly {num_queries} values, received {mask.shape[0]}"
            )
        mask = mask.unsqueeze(0).expand(batch_size, -1)
    elif mask.ndim == 2:
        if mask.shape != (batch_size, num_queries):
            raise ValueError(
                f"explicit_token_masking must have shape [{num_queries}] or "
                f"[batch, {num_queries}], received {tuple(mask.shape)}"
            )
    else:
        raise ValueError(
            f"explicit_token_masking must have shape [{PILOT_TOTAL_QUERY_COUNT}] or "
            f"[batch, {PILOT_TOTAL_QUERY_COUNT}], received {tuple(mask.shape)}"
        )

    if not torch.isfinite(mask).all():
        raise ValueError("explicit_token_masking values must be finite")
    if torch.any((mask < 0.0) | (mask > 1.0)):
        raise ValueError("explicit_token_masking values must stay within [0.0, 1.0]")
    return mask


def _is_flat_string_sequence(value: object) -> bool:
    if isinstance(value, str):
        return False
    if not isinstance(value, Sequence):
        return False
    return all(isinstance(item, str) for item in value)


def _normalize_empty_strength_spec(value: object) -> object | None:
    if value is None:
        return None
    if isinstance(value, Tensor):
        if value.ndim == 1 and value.shape[0] == 0:
            return None
        return value
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and len(value) == 0:
        return None
    return value


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


__all__ = [
    "AugmentedConditioning",
    "ComPhoserQFormer",
    "ComPhoserQFormerOutput",
    "DEFAULT_QFORMER_COND_SUMMARY_TOKENS",
    "DEFAULT_QFORMER_NUM_LAYERS",
    "DEFAULT_QFORMER_QUERY_COUNT",
    "DEFAULT_QFORMER_TASK_ID",
    "QFORMER_CONTROLLER_LAYOUT_PROMPT_IMAGE_ROUTER_V3",
    "QFORMER_CONTROLLER_LAYOUT_PROMPT_ROUTER_V2",
    "append_query_tokens_to_prompt",
    "build_batch_query_gate_target_mask",
    "build_query_gate_target_mask",
    "build_synthetic_txt_ids",
]

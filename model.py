import time
import torch
from types import SimpleNamespace
from typing import Callable, Optional
from typing_extensions import Unpack
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3MLP,
    GradientCheckpointingLayer,
    FlashAttentionKwargs,
    rotate_half,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache

INFER_POS_DEBUG = False

# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return torch.cat(selected_states, dim=-1)


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size) / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def _cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _resolve_attr(obj: object, path: tuple[str, ...]):
    cur = obj
    for name in path:
        if not hasattr(cur, name):
            return None
        cur = getattr(cur, name)
    return cur


def _get_embed_tokens(target: nn.Module):
    if hasattr(target, "get_input_embeddings"):
        embed_tokens = target.get_input_embeddings()
        if embed_tokens is not None:
            return embed_tokens

    candidate_paths = (
        ("model", "embed_tokens"),
        ("language_model", "embed_tokens"),
        ("language_model", "model", "embed_tokens"),
        ("model", "language_model", "embed_tokens"),
        ("model", "model", "embed_tokens"),
    )
    for path in candidate_paths:
        embed_tokens = _resolve_attr(target, path)
        if embed_tokens is not None:
            return embed_tokens
    raise AttributeError(f"Cannot find embed_tokens in {type(target).__name__}")


def _get_lm_head(target: nn.Module):
    if hasattr(target, "get_output_embeddings"):
        lm_head = target.get_output_embeddings()
        if lm_head is not None:
            return lm_head

    candidate_paths = (
        ("lm_head",),
        ("language_model", "lm_head"),
        ("model", "lm_head"),
        ("model", "language_model", "lm_head"),
    )
    for path in candidate_paths:
        lm_head = _resolve_attr(target, path)
        if lm_head is not None:
            return lm_head
    raise AttributeError(f"Cannot find lm_head in {type(target).__name__}")


def _build_infer_position_plan(
    *,
    model: "DFlashDraftModel",
    target: nn.Module,
    input_ids: torch.LongTensor,
    output_len: int,
    image_grid_thw: Optional[torch.Tensor],
    video_grid_thw: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      - prefill_pos_ids: shape [3, 1, num_input_tokens] for target prefill when MRoPE is enabled,
                        or shape [1, num_input_tokens] for 1D RoPE.
      - full_pos_ids:    shape [3, output_len] when MRoPE is enabled,
                        or shape [output_len] for 1D RoPE.
    """
    num_input_tokens = int(input_ids.shape[1])
    use_mrope = bool(getattr(model, "use_mrope", False))

    if use_mrope and hasattr(target, "model") and hasattr(target.model, "get_rope_index"):
        attention_mask = torch.ones_like(input_ids, device=input_ids.device, dtype=torch.long)
        prefill_pos_ids, _ = target.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        prefill_pos_ids = prefill_pos_ids.to(input_ids.device)

        full_pos_ids = torch.zeros((3, output_len), dtype=prefill_pos_ids.dtype, device=input_ids.device)
        full_pos_ids[:, :num_input_tokens] = prefill_pos_ids[:, 0, :]

        next_text_pos = int(prefill_pos_ids[:, 0, :].max().item()) + 1
        if output_len > num_input_tokens:
            tail = torch.arange(
                next_text_pos,
                next_text_pos + (output_len - num_input_tokens),
                device=input_ids.device,
                dtype=prefill_pos_ids.dtype,
            )
            full_pos_ids[:, num_input_tokens:] = tail.unsqueeze(0).expand(3, -1)
        return prefill_pos_ids, full_pos_ids

    prefill_pos_1d = torch.arange(num_input_tokens, device=input_ids.device).unsqueeze(0)
    full_pos_1d = torch.arange(output_len, device=input_ids.device)
    return prefill_pos_1d, full_pos_1d


def _ensure_rope_parameters(config: Qwen3Config) -> None:
    default_mrope = [24, 20, 20]
    default_theta = getattr(config, "rope_theta", 1000000)

    rope_params = getattr(config, "rope_parameters", None) or {}
    rope_scaling = getattr(config, "rope_scaling", None) or {}

    rope_type = rope_params.get("rope_type", rope_scaling.get("rope_type", "default"))
    mrope_section = rope_params.get("mrope_section", rope_scaling.get("mrope_section", default_mrope))
    rope_theta = rope_params.get("rope_theta", rope_scaling.get("rope_theta", default_theta))

    # Newer Qwen3 code path reads `rope_parameters`.
    rope_params["rope_type"] = rope_type
    rope_params["mrope_section"] = mrope_section
    rope_params["rope_theta"] = rope_theta
    config.rope_parameters = rope_params

    # Qwen3-VL rotary in transformers 4.57.x reads `rope_scaling`.
    rope_scaling["rope_type"] = rope_type
    rope_scaling["mrope_section"] = mrope_section
    rope_scaling["rope_theta"] = rope_theta
    # Some utility code expects `type`.
    rope_scaling["type"] = rope_scaling.get("type", rope_type)
    config.rope_scaling = rope_scaling


@torch.inference_mode()
def dflash_generate(
    model: "DFlashDraftModel",
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    temperature: float,
    pixel_values: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
    mask_token_id: Optional[int] = None,
    return_stats: bool = False,
):
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = model.block_size if block_size is None else block_size
    mask_token_id = model.mask_token_id if mask_token_id is None else mask_token_id

    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id, dtype=torch.long, device=target.device,
    )
    embed_tokens = _get_embed_tokens(target)
    lm_head = _get_lm_head(target)
    prefill_pos_ids, full_pos_ids = _build_infer_position_plan(
        model=model,
        target=target,
        input_ids=input_ids,
        output_len=output_ids.shape[1],
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
    )
    use_mrope = bool(getattr(model, "use_mrope", False))
    if return_stats and INFER_POS_DEBUG:
        prefill_shape = tuple(prefill_pos_ids.shape)
        if use_mrope:
            first_generated_pos = full_pos_ids[:, num_input_tokens].detach().cpu().tolist()
        else:
            first_generated_pos = int(full_pos_ids[num_input_tokens].item())
        print(
            f"[InferPos] infer_use_mrope={use_mrope} | "
            f"prefill_pos_shape={prefill_shape} | "
            f"first_generated_pos={first_generated_pos}"
        )
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    stop_ids_tensor = None
    if stop_token_ids is not None:
        stop_ids_tensor = torch.tensor(stop_token_ids, device=target.device, dtype=torch.long)

    prefill_start = _cuda_time() if return_stats else None
    target_kwargs = {}
    if pixel_values is not None:
        target_kwargs["pixel_values"] = pixel_values
        target_kwargs["image_grid_thw"] = image_grid_thw
    if pixel_values_videos is not None:
        target_kwargs["pixel_values_videos"] = pixel_values_videos
        target_kwargs["video_grid_thw"] = video_grid_thw

    output = target(
        input_ids,
        position_ids=prefill_pos_ids,
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=block_size > 1,
        **target_kwargs,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
    time_to_first_token = _cuda_time() - prefill_start if return_stats else None

    decode_start = _cuda_time() if return_stats else None
    acceptance_lengths = []
    start = num_input_tokens
    draft_prefill = True

    while start < max_length:
        prev_start = start
        block_output_ids = output_ids[:, start : start + block_size].clone()
        if use_mrope:
            block_position_ids = full_pos_ids[:, start : start + block_size].unsqueeze(1)
        else:
            block_position_ids = full_pos_ids[start : start + block_size].unsqueeze(0)
        if block_size > 1:
            noise_embedding = embed_tokens(block_output_ids)
            if use_mrope:
                draft_position_ids = full_pos_ids[:, past_key_values_draft.get_seq_length(): start + block_size].unsqueeze(1)
            else:
                draft_position_ids = full_pos_ids[past_key_values_draft.get_seq_length(): start + block_size].unsqueeze(0)
            expected_kv_len = target_hidden.shape[1] + noise_embedding.shape[1]
            draft_cache_position = None
            if draft_position_ids.shape[-1] == expected_kv_len:
                draft_cache_position = draft_position_ids[0, 0] if use_mrope else draft_position_ids[0]
            draft_logits = lm_head(model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=draft_position_ids,
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
                cache_position=draft_cache_position,
            )[:, 1 - block_size :, :])
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)
            if draft_prefill and return_stats:
                draft_prefill = False
                decode_start = _cuda_time()

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=block_size > 1,
        )

        posterior = sample(output.logits, temperature)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        acceptance_lengths.append(acceptance_length + 1)

        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :acceptance_length + 1, :]

        if stop_ids_tensor is not None:
            # Check only tokens appended in this iteration:
            # accepted prefix [prev_start : prev_start + acceptance_length]
            # plus posterior token at [prev_start + acceptance_length + 1].
            just_appended = output_ids[0, prev_start : prev_start + acceptance_length + 2]
            if torch.isin(just_appended, stop_ids_tensor).any():
                break

    output_ids = output_ids[:, :min(start + 1, max_length)]
    if stop_token_ids is not None:
        stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    if not return_stats:
        return output_ids

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = _cuda_time() - decode_start
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode_time / num_output_tokens,
        acceptance_lengths=acceptance_lengths,
    )


# ---------------------------------------------------------------------------
# DFlash model
# ---------------------------------------------------------------------------

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        _ensure_rope_parameters(config)
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.target_layer_ids = self.config.dflash_config.get(
            "target_layer_ids", build_target_layer_ids(config.num_target_layers, config.num_hidden_layers)
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        use_mrope = bool(self.config.dflash_config.get("use_mrope", True))
        self.use_mrope = use_mrope
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config) if use_mrope else Qwen3RotaryEmbedding(config)
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_params = getattr(config, "rope_parameters", None)
        print(
            f"[Init][DraftRoPE] use_mrope={self.use_mrope} | "
            f"rope_scaling={rope_scaling} | rope_parameters={rope_params}"
        )
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = self.config.dflash_config.get("mask_token_id", None)
        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ):
        self.eval()
        return dflash_generate(
            self,
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )

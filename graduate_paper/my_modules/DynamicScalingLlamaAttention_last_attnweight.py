import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from transformers.models.llama.modeling_llama import LlamaConfig, Cache, rotate_half, repeat_kv
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
import json
import os

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



class DynamicScalingLlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self._init_rope()

        self.current_scale_map = None # 今回の検索スコアのマップ、外部から強制的に値を注入
        self.attn_weights_last = None

    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(dim=self.head_dim)

    def _compute_attention(self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout_p: float = 0.0,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scaled Dot-Product Attentionを計算し、整形された出力を返します。
        """

        # print(f"DEBUG: query_states shape: {query_states.shape}")
        # print(f"DEBUG: key_states shape: {key_states.shape}")
        # if attention_mask is not None:
        #     print(f"DEBUG: attention_mask shape: {attention_mask.shape}")

        bsz, num_heads, q_len, head_dim = query_states.size()
        kv_seq_len = key_states.shape[-2]

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

        if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
            raise ValueError(f"Attention weights size mismatch: {attn_weights.size()}")

        # マスクの適用
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(f"Attention mask size mismatch: {attention_mask.size()}")
            attn_weights = attn_weights + attention_mask

        # Softmax (精度向上のため fp32 で計算)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout_p, training=training)

        value_states = value_states.contiguous()

        attn_output = torch.matmul(attn_weights, value_states.contiguous())
        if attn_output.size() != (bsz, num_heads, q_len, head_dim):
            raise ValueError(f"attn_output size mismatch: {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        return attn_output, attn_weights

    def _init_hidden_states_scale(self, hidden_states):
        last_recompute_tokens = 1

        bsz, q_len, _ = hidden_states.size()
        scale_map = getattr(self.config, "current_scale_map", None)

        hidden_states_scaled = hidden_states.clone()
        if scale_map is not None and self.layer_idx in self.config.hidden_scale_config["target_layers"]:
            current_scales = scale_map[:, :q_len].to(device=hidden_states.device, dtype=hidden_states.dtype)
            for dim in self.config.hidden_scale_config["target_dims"]:
                hidden_states_scaled[:, :, dim] = hidden_states[:, :, dim] * current_scales

        query_states_scaled = self.q_proj(hidden_states_scaled[:, -last_recompute_tokens:,:])
        key_states_scaled = self.k_proj(hidden_states_scaled)
        value_states = self.v_proj(hidden_states)

        query_states_scaled = query_states_scaled.view(bsz, last_recompute_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        key_states_scaled = key_states_scaled.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        return query_states_scaled, key_states_scaled, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        query_states_scaled,key_states_scaled,value_states_scaled = self._init_hidden_states_scale(hidden_states)
        last_recompute_tokens = 1
        use_cache = True if past_key_values is not None else False

        #in the generation, the last token is always recomputed
        if q_len==1 and use_cache:
            # print("decoding")
            # print(f"DEBUG: Layer {self.layer_idx} | q_len: {q_len} | past_exists: {past_key_values is not None}")
            # print("use_cache: ",use_cache)
            query_states = query_states_scaled
            key_states = key_states_scaled
            if value_states_scaled is None:
                value_states = self.v_proj(hidden_states)
                value_states=value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            else:
                value_states=value_states_scaled

            kv_seq_len = key_states.shape[-2]
            if past_key_values is not None:
                if self.layer_idx is None:
                    raise ValueError(
                        f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                        "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                        "with a layer index."
                    )
                kv_seq_len += past_key_values.get_seq_length(self.layer_idx)


            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
                # print(f"DEBUG: Layer {self.layer_idx} Updating cache. Input K: {key_states.shape}")
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
                # print(f"DEBUG: Layer {self.layer_idx} Updated cache. Output K: {key_states.shape}")

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_output, attn_weights = self._compute_attention(query_states, key_states, value_states, attention_mask, self.attention_dropout, training=self.training)

        else: # Prefill
            # print("prefill")
            # print(f"DEBUG: Layer {self.layer_idx} | q_len: {q_len} | past_exists: {past_key_values is not None}")
            # print("use_cache: ",use_cache)

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            if value_states_scaled is None:
                value_states_scaled = value_states

            kv_seq_len = key_states.shape[-2]
            if past_key_values is not None:
                if self.layer_idx is None:
                    raise ValueError(
                        f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                        "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                        "with a layer index."
                    )
                kv_seq_len += past_key_values.get_seq_length(self.layer_idx)

            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            _, key_states_scaled = apply_rotary_pos_emb(key_states_scaled, key_states_scaled, cos, sin, position_ids)
            #apply rotary position embedding to query_states which is scaled

            position_ids_for_query = position_ids[:,-last_recompute_tokens:]
            query_states_scaled,_ = apply_rotary_pos_emb(query_states_scaled, query_states_scaled, cos, sin, position_ids_for_query)

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
                # scaleしたKV cacheを保存
                key_states, value_states = past_key_values.update(key_states_scaled, value_states_scaled, self.layer_idx, cache_kwargs)

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            key_states_scaled = repeat_kv(key_states_scaled, self.num_key_value_groups)
            value_states_scaled = repeat_kv(value_states_scaled, self.num_key_value_groups)

            attn_output, attn_weights = self._compute_attention(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                dropout_p=self.attention_dropout,
                training=self.training
                )

            # --- 抽出用に追加 ---
            self.last_q_scaled = query_states_scaled.detach().cpu() 
            self.last_k_scaled = key_states_scaled.detach().cpu()
            # ------------------

            #crop attention_mask
            attention_mask_last = attention_mask[:,:, -last_recompute_tokens:,:]


            attn_output_last, attn_weights_last = self._compute_attention(
                query_states=query_states_scaled,
                key_states=key_states_scaled,
                value_states=value_states_scaled,
                attention_mask=attention_mask_last,
                dropout_p=self.attention_dropout,
                training=self.training
            )

            # --- 抽出用に追加 ---
            self.attn_weights_last = attn_weights_last
            # ------------------

            #compute attention weights of the last row
            attn_weights_last = torch.matmul(query_states_scaled, key_states_scaled.transpose(2, 3)) / math.sqrt(self.head_dim)

            attn_weights_last=attn_weights_last+attention_mask_last

            attn_weights_last = nn.functional.softmax(attn_weights_last, dim=-1, dtype=torch.float32).to(query_states.dtype)

            attn_weights_last = nn.functional.dropout(attn_weights_last, p=self.attention_dropout, training=self.training)

            #compute attention output of the last row
            attn_output_last = torch.matmul(attn_weights_last, value_states_scaled)

            attn_output_last = attn_output_last.transpose(1, 2).contiguous()

            attn_output_last = attn_output_last.reshape(bsz, last_recompute_tokens, self.hidden_size)
            #最後の一行を変える
            attn_output[:, -last_recompute_tokens:,:] = attn_output_last

        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights
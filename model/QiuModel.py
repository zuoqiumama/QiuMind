import math
from typing import List, Optional, Tuple, Union

from transformers import PretrainedConfig
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

class QiuMindConfig(PretrainedConfig):
    model_type = "qiumind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def _norm(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return x

    def forward(self, x):
        x = self._norm(x)
        return x * self.weight


def precompute_freqs(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    # 1. 初始化标准 RoPE 频率。
    # torch.arange(0, dim, 2) 生成 [0, 2, 4, ... dim-2]
    # 计算出的 freqs 就是标准的 1 / (base ** (2i / d))
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )

    if rope_scaling is not None:
        # 2. 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        # 只有当要推断的长度大于原始训练长度时，才应用缩放
        if end / orig_max > 1.0:
            # 3. 使用前文推导的公式，定义波长比例 b 到维度索引 i 的映射函数
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                2 * math.log(rope_base)
            )

            # 4. 计算高频区和低频区的维度切分点
            # low: 不需要缩放的高频部分的最高索引
            # high: 需要完全缩放的低频部分的最低索引
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            # 5. 计算混合因子 γ (Ramp)
            # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
            # clamp 函数限制了数值只能在 [0, 1] 之间。
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )

            # 6. 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
            # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
            # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
            # ramp在0-1之间时：平滑过渡。
            freqs = freqs * (1 - ramp + ramp / factor)

    # 7. 根据目标长度 end，生成位置索引向量 t
    t = torch.arange(end, device=freqs.device)

    # 8. 计算外积：将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
    freqs = torch.outer(t, freqs).float()

    # 9. 计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


_ROPE_CACHE = {}


def _rope_cache_key(dim, end, rope_base, rope_scaling, device):
    rope_scaling_key = None
    if rope_scaling is not None:
        rope_scaling_key = tuple(sorted(rope_scaling.items()))
    return (dim, end, rope_base, rope_scaling_key, device.type, device.index)


def get_rope_cache(dim, end, rope_base, rope_scaling, device):
    key = _rope_cache_key(dim, end, rope_base, rope_scaling, device)
    if key not in _ROPE_CACHE:
        freqs_cos, freqs_sin = precompute_freqs(
            dim=dim,
            end=end,
            rope_base=rope_base,
            rope_scaling=rope_scaling,
        )
        _ROPE_CACHE[key] = (freqs_cos.to(device), freqs_sin.to(device))
    return _ROPE_CACHE[key]

class Attention(nn.Module):
    def __init__(self, config: QiuMindConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scale = self.head_dim**-0.5

        assert (
            config.hidden_size % config.num_attention_heads == 0
        ), "hidden_size must be divisible by num_attention_heads"
        assert (
            config.num_attention_heads % config.num_key_value_heads == 0
        ), "num_attention_heads must be divisible by num_key_value_heads"

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _apply_rope(self, q, k, cos, sin):
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

        def rotate_half(x):
            return torch.cat(
                (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
            )

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = hidden_states.size()
        past_len = 0

        if past_key_value is not None:
            past_len = past_key_value[0].size(1)

        q = self.q_proj(hidden_states).view(
            bsz, seq_len, self.num_attention_heads, self.head_dim
        )
        k = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        )

        if position_embeddings is None:
            raise ValueError("position_embeddings must be provided by QiuMindModel")

        cos, sin = position_embeddings
        q_embed, k_embed = self._apply_rope(q, k, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k_embed = torch.cat([past_k, k_embed], dim=1)
            v = torch.cat([past_v, v], dim=1)

        present_key_value = (k_embed, v) if use_cache else None

        k_embed = repeat_kv(k_embed, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        q_embed = q_embed.transpose(1, 2)
        k_embed = k_embed.transpose(1, 2)
        v = v.transpose(1, 2)

        k_len = k_embed.size(-2)
        causal_mask = None
        if attention_mask is not None or past_len > 0:
            q_pos = torch.arange(seq_len, device=hidden_states.device) + past_len
            k_pos = torch.arange(k_len, device=hidden_states.device)
            causal_forbidden = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
            causal_mask = torch.zeros(
                (seq_len, k_len), device=hidden_states.device, dtype=q_embed.dtype
            )
            causal_mask.masked_fill_(causal_forbidden, torch.finfo(q_embed.dtype).min)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        attn_bias = None
        if attention_mask is not None and causal_mask is not None:
            attn_bias = attention_mask + causal_mask
        elif attention_mask is not None:
            attn_bias = attention_mask
        elif causal_mask is not None:
            attn_bias = causal_mask

        if self.config.flash_attention:
            attn_output = F.scaled_dot_product_attention(
                q_embed,
                k_embed,
                v,
                attn_mask=attn_bias,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=attn_bias is None,
                scale=self.scale,
            )
        else:
            attn_weights = torch.matmul(q_embed * self.scale, k_embed.transpose(-2, -1))

            if attn_bias is not None:
                attn_weights += attn_bias
            elif past_len == 0:
                q_pos = torch.arange(seq_len, device=hidden_states.device)
                k_pos = torch.arange(k_len, device=hidden_states.device)
                causal_forbidden = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
                attn_weights = attn_weights.masked_fill(
                    causal_forbidden.unsqueeze(0).unsqueeze(0),
                    torch.finfo(attn_weights.dtype).min,
                )

            attn_probs = torch.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
            attn_output = torch.matmul(attn_probs, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output = self.out_proj(attn_output)
        return attn_output, present_key_value

    
class FeedForward(nn.Module):
    def __init__(self, config: QiuMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))

class QiuMindBlock(nn.Module):
    def __init__(self, config: QiuMindConfig):
        super().__init__()
        self.attn = Attention(config)
        self.ffn = FeedForward(config)
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_output, present_key_value = self.attn(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attn_output
        ffn_output = self.ffn(self.norm2(hidden_states))
        hidden_states = hidden_states + ffn_output
        return hidden_states, present_key_value
    

class QiuMindModel(nn.Module):
    def __init__(self, config: QiuMindConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([QiuMindBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]]:
        hidden_states = self.embed_tokens(input_ids)
        bsz, seq_len, _ = hidden_states.size()

        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            past_len = past_key_values[0][0].size(1)

        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + seq_len, device=hidden_states.device
            ).unsqueeze(0)

        if position_ids.size(0) == 1 and bsz > 1:
            position_ids = position_ids.expand(bsz, -1)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                if past_len > 0 and attention_mask.size(1) == seq_len:
                    past_attention = torch.ones(
                        (bsz, past_len),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat([past_attention, attention_mask], dim=1)

                attention_mask = attention_mask[:, None, None, :].to(hidden_states.dtype)
                attention_mask = (
                    1.0 - attention_mask
                ) * torch.finfo(hidden_states.dtype).min
            elif attention_mask.dim() == 4:
                attention_mask = attention_mask.to(hidden_states.dtype)
            else:
                raise ValueError(
                    f"Unsupported attention_mask shape: {attention_mask.shape}"
                )

        freqs_cos, freqs_sin = get_rope_cache(
            dim=self.config.hidden_size // self.config.num_attention_heads,
            end=self.config.max_position_embeddings,
            rope_base=self.config.rope_theta,
            rope_scaling=self.config.rope_scaling,
            device=hidden_states.device,
        )
        cos = freqs_cos[position_ids]
        sin = freqs_sin[position_ids]
        position_embeddings = (cos, sin)

        all_present_key_values = () if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past_key_value = past_key_values[i] if past_key_values is not None else None
            hidden_states, present_key_value = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_value=layer_past_key_value,
                use_cache=use_cache,
            )
            if use_cache:
                all_present_key_values += (present_key_value,)

        hidden_states = self.norm(hidden_states)
        return hidden_states, all_present_key_values



class QiuMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = QiuMindConfig

    def __init__(self, config: QiuMindConfig):
        super().__init__(config)
        self.model = QiuMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 优化1：权重共享 (假设 QiuMindModel 内部有 embed_tokens)
        if hasattr(self.model, "embed_tokens"):
            self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,  # 新增：用于显存优化
        **args,
    ) -> CausalLMOutputWithPast:
        
        # 获取底层模型的输出
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )

        # 兼容处理：QiuMindModel 原来只返回 hidden_states 和 past_key_values
        hidden_states = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs
        present_key_values = model_outputs[1] if isinstance(model_outputs, tuple) and len(model_outputs) > 1 else None

        # 优化2：仅计算需要的 logits，避免显存溢出
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # 优化3：支持传入 labels 计算交叉熵损失 (训练必备)
        loss = None
        if labels is not None:
            # Shift logits and labels 使得预测对齐
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # 优化4：返回标准的 HuggingFace CausalLMOutputWithPast 格式
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=present_key_values,
            hidden_states=hidden_states,
        )

        # 兼容 MoE 架构可能输出的辅助损失
        if isinstance(model_outputs, tuple) and len(model_outputs) > 2:
            output.aux_loss = model_outputs[2]

        return output
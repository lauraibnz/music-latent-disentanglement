import math

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import SinusoidalPositionalEmbedding


def _get_valid_heads(channels, max_heads):
    for heads in range(min(int(max_heads), int(channels)), 0, -1):
        if channels % heads == 0:
            return heads
    return 1


def _build_sinusoidal_positions(length, dim, device, dtype):
    half_dim = dim // 2
    if half_dim == 0:
        return torch.zeros(1, length, dim, device=device, dtype=dtype)
    positions = torch.arange(length, device=device, dtype=dtype)
    exponent = -math.log(10000.0) / max(half_dim - 1, 1)
    freqs = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * exponent)
    angles = positions[:, None] * freqs[None, :]
    pos = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if dim % 2 == 1:
        pos = F.pad(pos, (0, 1))
    return pos.unsqueeze(0)


def _build_rope_freqs(dim, max_len=4096, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rope_qk(x, freqs):
    if freqs is None:
        return x

    x = x.contiguous()
    seq_len = x.shape[2]
    rot_dim = min(freqs.shape[-1] * 2, x.shape[-1])
    rot_dim = rot_dim - (rot_dim % 2)
    if rot_dim <= 0:
        return x

    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]

    x_complex = torch.view_as_complex(x_rot.float().reshape(*x_rot.shape[:-1], -1, 2))
    rope = freqs[:seq_len].to(device=x.device)
    x_rotated = torch.view_as_real(x_complex * rope.unsqueeze(0).unsqueeze(0)).flatten(-2)
    x_rotated = x_rotated.type_as(x)

    if x_pass.numel() == 0:
        return x_rotated
    return torch.cat([x_rotated, x_pass], dim=-1)


class FeedForward(nn.Module):
    def __init__(self, model_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(model_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TimbreProjection(nn.Module):
    """Project timbre and diffusion time into a shared conditioning vector."""

    def __init__(self, timbre_dim, time_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(int(timbre_dim) + int(time_dim), int(output_dim) * 2)

        self.net = nn.Sequential(
            nn.Linear(int(timbre_dim) + int(time_dim), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, int(output_dim)),
        )

    def forward(self, timbre_emb, time_emb):
        return self.net(torch.cat([timbre_emb, time_emb], dim=-1))


class TransformerBlock1D(nn.Module):
    """DiT-style block with timbre conditioning and structure cross-attention."""

    def __init__(
        self,
        model_dim,
        cond_dim=0,
        time_cond_dim=0,
        use_cond_in_mlp=False,
        use_time_cond_cross_attention_rope=False,
        use_time_cond_cross_attention_correction=False,
        time_cond_correction_scale=0.05,
        time_cond_gate_init_bias=-5.0,
        attention_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        use_adaLN_zero=True,
    ):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.time_cond_dim = int(time_cond_dim)
        self.use_adaLN_zero = bool(use_adaLN_zero)
        self.use_cond_in_mlp = bool(use_cond_in_mlp)
        self.use_time_cond_cross_attention_rope = bool(use_time_cond_cross_attention_rope)
        self.use_time_cond_cross_attention_correction = bool(use_time_cond_cross_attention_correction)
        self.time_cond_correction_scale = float(time_cond_correction_scale)
        self.num_heads = int(attention_heads)
        if model_dim % self.num_heads != 0:
            raise ValueError(
                f"model_dim={model_dim} must be divisible by attention_heads={self.num_heads}"
            )
        self.head_dim = model_dim // self.num_heads
        self.attention_dropout_p = float(attention_dropout)

        affine = (self.cond_dim == 0)
        self.attn_norm = nn.LayerNorm(model_dim, elementwise_affine=affine)
        self.cross_attn_norm = nn.LayerNorm(model_dim, elementwise_affine=affine)
        self.mlp_norm = nn.LayerNorm(model_dim, elementwise_affine=affine)

        self.qkv_proj = nn.Linear(model_dim, model_dim * 3)
        self.attn_out = nn.Linear(model_dim, model_dim)
        self.attn_dropout = nn.Dropout(dropout)

        if self.time_cond_dim > 0:
            self.cross_q_proj = nn.Linear(model_dim, model_dim)
            self.cross_k_proj = nn.Linear(self.time_cond_dim, model_dim)
            self.cross_v_proj = nn.Linear(self.time_cond_dim, model_dim)
            self.cross_attn_out = nn.Linear(model_dim, model_dim)
            self.cross_attn_dropout = nn.Dropout(dropout)
            nn.init.zeros_(self.cross_attn_out.weight)
            nn.init.zeros_(self.cross_attn_out.bias)
        else:
            self.cross_q_proj = None
            self.cross_k_proj = None
            self.cross_v_proj = None
            self.cross_attn_out = None
            self.cross_attn_dropout = None

        if self.time_cond_dim > 0 and self.use_time_cond_cross_attention_correction:
            self.cross_attn_struct_corr = nn.Linear(model_dim + self.time_cond_dim, model_dim)
            nn.init.zeros_(self.cross_attn_struct_corr.weight)
            nn.init.zeros_(self.cross_attn_struct_corr.bias)
            if self.cond_dim > 0:
                self.cross_attn_time_gate = nn.Linear(self.cond_dim, model_dim)
                self.cross_attn_time_gate_param = None
                nn.init.zeros_(self.cross_attn_time_gate.weight)
                nn.init.constant_(self.cross_attn_time_gate.bias, float(time_cond_gate_init_bias))
            else:
                self.cross_attn_time_gate = None
                self.cross_attn_time_gate_param = nn.Parameter(
                    torch.full((model_dim,), float(time_cond_gate_init_bias))
                )
        else:
            self.cross_attn_struct_corr = None
            self.cross_attn_time_gate = None
            self.cross_attn_time_gate_param = None

        self.mlp = FeedForward(model_dim=model_dim, mlp_ratio=mlp_ratio, dropout=dropout)

        if self.cond_dim > 0:
            if self.use_adaLN_zero:
                self.adaLN_modulation = nn.Linear(self.cond_dim, model_dim * 6)
                nn.init.zeros_(self.adaLN_modulation.weight)
                nn.init.zeros_(self.adaLN_modulation.bias)
                self.attn_mod = None
                self.mlp_mod = None
            else:
                self.adaLN_modulation = None
                self.attn_mod = nn.Linear(self.cond_dim, model_dim * 2)
                self.mlp_mod = nn.Linear(self.cond_dim, model_dim * 2)
                nn.init.zeros_(self.attn_mod.weight)
                nn.init.zeros_(self.attn_mod.bias)
                nn.init.zeros_(self.mlp_mod.weight)
                nn.init.zeros_(self.mlp_mod.bias)
        else:
            self.adaLN_modulation = None
            self.attn_mod = None
            self.mlp_mod = None

        if self.cond_dim > 0 and self.use_cond_in_mlp:
            self.mlp_cond = nn.Linear(self.cond_dim, model_dim)
        else:
            self.mlp_cond = None

    @staticmethod
    def _modulate(x, scale, shift):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def _attention(self, x, rope_freqs=None):
        bsz, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope_qk(q, rope_freqs)
        k = _apply_rope_qk(k, rope_freqs)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attention_dropout_p if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.attn_out(attn)

    def _cross_attention(self, x, memory, rope_freqs=None):
        bsz, seq_len, _ = x.shape
        mem_len = memory.shape[1]

        q = self.cross_q_proj(x)
        k = self.cross_k_proj(memory)
        v = self.cross_v_proj(memory)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, mem_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, mem_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_time_cond_cross_attention_rope:
            q = _apply_rope_qk(q, rope_freqs)
            k = _apply_rope_qk(k, rope_freqs)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attention_dropout_p if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.cross_attn_out(attn)

    def _apply_cross_attention_correction(self, x, time_cond, cond):
        if time_cond is None or self.cross_attn_struct_corr is None:
            return x

        if time_cond.shape[1] != x.shape[1]:
            time_cond = F.interpolate(
                time_cond.transpose(1, 2),
                size=x.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        struct_corr = self.cross_attn_struct_corr(torch.cat([x, time_cond], dim=-1))
        if cond is not None and self.cross_attn_time_gate is not None:
            gate = torch.sigmoid(self.cross_attn_time_gate(cond)).unsqueeze(1)
        elif self.cross_attn_time_gate_param is not None:
            gate = torch.sigmoid(self.cross_attn_time_gate_param).view(1, 1, -1)
        else:
            return x
        return x + self.time_cond_correction_scale * gate * struct_corr

    def forward(self, x, cond=None, time_cond=None, rope_freqs=None):
        if self.use_adaLN_zero and cond is not None and self.adaLN_modulation is not None:
            scale_attn, shift_attn, gate_attn, scale_mlp, shift_mlp, gate_mlp = (
                self.adaLN_modulation(cond).chunk(6, dim=-1)
            )

            attn_input = self._modulate(self.attn_norm(x), scale_attn, shift_attn)
            attn_output = self._attention(attn_input, rope_freqs=rope_freqs)
            x = x + gate_attn.unsqueeze(1) * self.attn_dropout(attn_output)

            if time_cond is not None and self.cross_attn_out is not None:
                cross_input = self.cross_attn_norm(x)
                cross_output = self._cross_attention(cross_input, time_cond, rope_freqs=rope_freqs)
                x = x + gate_attn.unsqueeze(1) * self.cross_attn_dropout(cross_output)
                x = self._apply_cross_attention_correction(x, time_cond, cond)

            mlp_input = self._modulate(self.mlp_norm(x), scale_mlp, shift_mlp)
            if self.mlp_cond is not None:
                mlp_input = mlp_input + self.mlp_cond(cond).unsqueeze(1)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)
            return x

        attn_input = self.attn_norm(x)
        if cond is not None and self.attn_mod is not None:
            scale, shift = self.attn_mod(cond).chunk(2, dim=-1)
            attn_input = self._modulate(attn_input, scale, shift)
        attn_output = self._attention(attn_input, rope_freqs=rope_freqs)
        x = x + self.attn_dropout(attn_output)

        if time_cond is not None and self.cross_attn_out is not None:
            cross_input = self.cross_attn_norm(x)
            cross_output = self._cross_attention(cross_input, time_cond, rope_freqs=rope_freqs)
            x = x + self.cross_attn_dropout(cross_output)
            x = self._apply_cross_attention_correction(x, time_cond, cond)

        mlp_input = self.mlp_norm(x)
        if cond is not None and self.mlp_mod is not None:
            scale, shift = self.mlp_mod(cond).chunk(2, dim=-1)
            mlp_input = self._modulate(mlp_input, scale, shift)
        if cond is not None and self.mlp_cond is not None:
            mlp_input = mlp_input + self.mlp_cond(cond).unsqueeze(1)
        x = x + self.mlp(mlp_input)
        return x


@gin.configurable
class DiffusionTransformer1D(nn.Module):
    """DiT-style transformer with global timbre conditioning and structure cross-attention."""

    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        model_dim=512,
        num_layers=8,
        mlp_ratio=4.0,
        cond_channels=512,
        time_cond_channels=512,
        time_embed_dim=128,
        attention_heads=8,
        dropout=0.1,
        attention_dropout=0.0,
        pos_embedding_type="rope",
        max_seq_len=2048,
        use_cond_input=False,
        use_cond_in_mlp=False,
        use_time_cond_cross_attention_rope=False,
        use_time_cond_cross_attention_correction=False,
        time_cond_correction_scale=0.05,
        time_cond_gate_init_bias=-5.0,
        use_adaLN_zero=True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.model_dim = int(model_dim)
        self.num_layers = int(num_layers)
        self.mlp_ratio = float(mlp_ratio)
        self.cond_channels = int(cond_channels)
        self.time_cond_channels = int(time_cond_channels)
        self.time_embed_dim = int(time_embed_dim)
        self.pos_embedding_type = str(pos_embedding_type).lower()
        self.max_seq_len = int(max_seq_len)
        self.use_cond_input = bool(use_cond_input)
        self.use_cond_in_mlp = bool(use_cond_in_mlp)
        self.use_time_cond_cross_attention_rope = bool(use_time_cond_cross_attention_rope)
        self.use_time_cond_cross_attention_correction = bool(use_time_cond_cross_attention_correction)
        self.time_cond_correction_scale = float(time_cond_correction_scale)
        self.time_cond_gate_init_bias = float(time_cond_gate_init_bias)
        self.use_adaLN_zero = bool(use_adaLN_zero)

        self.num_heads = _get_valid_heads(self.model_dim, attention_heads)
        self.head_dim = self.model_dim // self.num_heads

        self.time_embedding = SinusoidalPositionalEmbedding(self.time_embed_dim)

        if self.cond_channels > 0:
            self.cond_projection = TimbreProjection(
                timbre_dim=self.cond_channels,
                time_dim=self.time_embed_dim,
                output_dim=self.cond_channels + self.time_embed_dim,
            )
            self.time_proj = None
            self.total_cond = self.cond_channels + self.time_embed_dim
        else:
            self.cond_projection = None
            self.time_proj = nn.Sequential(
                nn.Linear(self.time_embed_dim, self.time_embed_dim),
                nn.SiLU(),
                nn.Linear(self.time_embed_dim, self.time_embed_dim),
            )
            self.total_cond = self.time_embed_dim

        self.input_proj = nn.Conv1d(self.in_channels, self.model_dim, kernel_size=1)
        self.output_norm = nn.LayerNorm(self.model_dim)
        self.output_proj = nn.Conv1d(self.model_dim, self.out_channels, kernel_size=1)
        if self.total_cond > 0 and self.use_cond_input:
            self.cond_input_proj = nn.Linear(self.total_cond, self.model_dim)
        else:
            self.cond_input_proj = None

        if self.time_cond_channels > 0:
            self.time_cond_proj = nn.Conv1d(self.time_cond_channels, self.model_dim, kernel_size=1)
        else:
            self.time_cond_proj = None

        if self.pos_embedding_type == "learned":
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_len, self.model_dim))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)
            self.rope_freqs = None
        elif self.pos_embedding_type == "rope":
            self.pos_embedding = None
            self.register_buffer(
                "rope_freqs",
                _build_rope_freqs(self.head_dim, max_len=self.max_seq_len * 2),
            )
        elif self.pos_embedding_type in {"sinusoidal", "none"}:
            self.pos_embedding = None
            self.rope_freqs = None
        else:
            raise ValueError(
                f"Unsupported pos_embedding_type='{pos_embedding_type}'. "
                "Expected one of {'learned', 'rope', 'sinusoidal', 'none'}."
            )

        block_time_cond_dim = self.model_dim if self.time_cond_channels > 0 else 0
        self.blocks = nn.ModuleList(
            [
                TransformerBlock1D(
                    model_dim=self.model_dim,
                    cond_dim=self.total_cond,
                    time_cond_dim=block_time_cond_dim,
                    use_cond_in_mlp=self.use_cond_in_mlp,
                    use_time_cond_cross_attention_rope=self.use_time_cond_cross_attention_rope,
                    use_time_cond_cross_attention_correction=self.use_time_cond_cross_attention_correction,
                    time_cond_correction_scale=self.time_cond_correction_scale,
                    time_cond_gate_init_bias=self.time_cond_gate_init_bias,
                    attention_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    use_adaLN_zero=self.use_adaLN_zero,
                )
                for _ in range(self.num_layers)
            ]
        )

        if self.use_adaLN_zero and self.total_cond > 0:
            self.final_modulation = nn.Linear(self.total_cond, self.model_dim * 2)
            nn.init.zeros_(self.final_modulation.weight)
            nn.init.zeros_(self.final_modulation.bias)
        else:
            self.final_modulation = None

    def _get_positional_embedding(self, seq_len, device, dtype):
        if self.pos_embedding_type == "none":
            return None
        if self.pos_embedding_type == "sinusoidal":
            return _build_sinusoidal_positions(seq_len, self.model_dim, device=device, dtype=dtype)
        if self.pos_embedding_type == "rope":
            return None
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input length {seq_len} exceeds configured max_seq_len={self.max_seq_len} "
                "for learned positional embeddings."
            )
        return self.pos_embedding[:, :seq_len].to(device=device, dtype=dtype)

    def _align_time_cond(self, time_cond_tokens, target_steps):
        if time_cond_tokens is None:
            return None
        if int(time_cond_tokens.shape[-1]) == int(target_steps):
            return time_cond_tokens
        return F.interpolate(time_cond_tokens, size=target_steps, mode="linear", align_corners=False)

    def forward(self, x, timesteps, cond=None, time_cond=None):
        original_size = x.shape[-1]
        t_emb = self.time_embedding(timesteps)

        if cond is not None and self.cond_projection is not None:
            global_cond = self.cond_projection(cond, t_emb)
        elif self.time_proj is not None:
            global_cond = self.time_proj(t_emb)
        else:
            global_cond = t_emb

        x = self.input_proj(x).transpose(1, 2)
        if global_cond is not None and self.cond_input_proj is not None:
            x = x + self.cond_input_proj(global_cond).unsqueeze(1)

        if self.pos_embedding_type != "rope":
            pos = self._get_positional_embedding(x.shape[1], device=x.device, dtype=x.dtype)
            if pos is not None:
                x = x + pos

        time_tokens = None
        if self.time_cond_proj is not None and time_cond is not None:
            time_cond = self.time_cond_proj(time_cond)
            time_cond = self._align_time_cond(time_cond, original_size)
            time_tokens = time_cond.transpose(1, 2)

        rope_freqs = self.rope_freqs if self.pos_embedding_type == "rope" else None
        for block in self.blocks:
            x = block(
                x,
                cond=global_cond,
                time_cond=time_tokens,
                rope_freqs=rope_freqs,
            )

        x = self.output_norm(x)
        if self.final_modulation is not None and global_cond is not None:
            scale, shift = self.final_modulation(global_cond).chunk(2, dim=-1)
            x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        x = self.output_proj(x.transpose(1, 2))
        if x.shape[-1] != original_size:
            x = F.interpolate(x, size=original_size, mode="linear", align_corners=False)
        return x

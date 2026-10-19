""" This file contains nets used for ManiFlow Policy. """
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp, RmsNorm

from robomimic.models.diffusion_policy_nets import SinusoidalPosEmb

# Adapted from ManiFlow DiTX (https://github.com/allenai/maniflow), itself adapted from
# DiT (https://github.com/facebookresearch/DiT) and RDT (https://github.com/thu-ml/RoboticsDiffusionTransformer).
# Language conditioning and modality pre-norm have been removed (RGB / low-dim only).


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, c):
        B, N, C = x.shape
        L = c.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(x)


class DiTXBlock(nn.Module):
    """
    Transformer block: adaLN-modulated self-attention over the action tokens,
    cross-attention to the observation tokens, then an MLP. The time embedding
    produces shift/scale/gate for each of the three sub-layers.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, p_drop_attn=0.0, qkv_bias=False, qk_norm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=p_drop_attn)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio),
                       act_layer=approx_gelu, drop=0.0)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size),
        )

    def forward(self, x, time_c, context_c):
        """
        x: [B, T, D] action tokens
        time_c: [B, D] time embedding
        context_c: [B, L, D] observation tokens
        """
        (shift_msa, scale_msa, gate_msa,
         shift_cross, scale_cross, gate_cross,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(time_c).chunk(9, dim=-1)

        h = modulate(self.norm1(x), shift_msa, scale_msa)
        h, _ = self.self_attn(h, h, h)
        x = x + gate_msa.unsqueeze(1) * h

        h = modulate(self.norm2(x), shift_cross, scale_cross)
        x = x + gate_cross.unsqueeze(1) * self.cross_attn(h, context_c)

        h = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.norm_final = RmsNorm(hidden_size, eps=1e-6)
        self.ffn_final = Mlp(in_features=hidden_size, hidden_features=hidden_size,
                             out_features=out_channels, act_layer=approx_gelu, drop=0)

    def forward(self, x):
        return self.ffn_final(self.norm_final(x))


class ManiFlowTransformer(nn.Module):
    """
    Consistency-flow denoiser with a DiT backbone (ManiFlow's DiTX).

    The network is conditioned on two scalars per sample, the flow time t and
    target_t (dt in "relative" mode, t + dt in "absolute" mode), which lets it
    predict average velocity rather than instantaneous velocity.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        horizon,
        cond_dim,
        cond_len,
        timestep_embed_dim=256,
        target_t_embed_dim=256,
        n_layer=3,
        n_head=4,
        n_emb=256,
        mlp_ratio=4.0,
        p_drop_attn=0.1,
        qkv_bias=False,
        qk_norm=False,
    ):
        """
        Args:
            input_dim (int): action dim
            output_dim (int): action dim
            horizon (int): number of predicted action steps (Tp)
            cond_dim (int): feature dim of each observation token
            cond_len (int): max number of observation tokens (observation_horizon for
                one token per frame)
        """
        super().__init__()
        self.horizon = horizon

        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.vis_cond_obs_emb = nn.Linear(cond_dim, n_emb)
        self.vis_cond_pos_embed = nn.Parameter(torch.zeros(1, cond_len, n_emb))

        self.flow_timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(timestep_embed_dim * 4, n_emb),
        )
        self.flow_target_t_encoder = nn.Sequential(
            SinusoidalPosEmb(target_t_embed_dim),
            nn.Linear(target_t_embed_dim, target_t_embed_dim * 4),
            nn.Mish(),
            nn.Linear(target_t_embed_dim * 4, n_emb),
        )
        self.timestep_target_t_adaptor = nn.Linear(n_emb * 2, n_emb)

        self.blocks = nn.ModuleList([
            DiTXBlock(n_emb, n_head, mlp_ratio=mlp_ratio, p_drop_attn=p_drop_attn,
                      qkv_bias=qkv_bias, qk_norm=qk_norm)
            for _ in range(n_layer)
        ])
        self.final_layer = FinalLayer(n_emb, output_dim)

        self.initialize_weights()

    def initialize_weights(self):
        for block in self.blocks:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            nn.init.zeros_(block.self_attn.in_proj_bias)
            nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
            nn.init.zeros_(block.self_attn.out_proj.bias)

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # zero-out adaLN modulation so every block starts as the identity
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.normal_(self.input_emb.weight, std=0.02)
        nn.init.constant_(self.input_emb.bias, 0)
        nn.init.normal_(self.pos_emb, std=0.02)

        for encoder in (self.flow_timestep_encoder, self.flow_target_t_encoder):
            for layer in encoder:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.02)
                    nn.init.constant_(layer.bias, 0)

        nn.init.normal_(self.vis_cond_obs_emb.weight, std=0.02)
        nn.init.constant_(self.vis_cond_obs_emb.bias, 0)
        nn.init.normal_(self.timestep_target_t_adaptor.weight, std=0.02)
        nn.init.constant_(self.timestep_target_t_adaptor.bias, 0)

        # zero-out the final output layer
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)

    @staticmethod
    def _to_batch_vector(x, B, like):
        # accepts python scalar, 0-dim tensor, or [B] tensor
        x = torch.as_tensor(x, dtype=like.dtype, device=like.device)
        return x.reshape(-1).expand(B)

    def forward(self, sample, timestep, target_t, global_cond):
        """
        Args:
            sample (torch.Tensor): noisy actions x_t, [B, T, input_dim]
            timestep (torch.Tensor or float): flow time t in [0, 1], [B] or scalar
            target_t (torch.Tensor or float): dt (relative) or t + dt (absolute), [B] or scalar
            global_cond (torch.Tensor): observation tokens, [B, L, cond_dim] with L <= cond_len

        Returns:
            velocity (torch.Tensor): [B, T, output_dim]
        """
        B = sample.shape[0]

        x = self.input_emb(sample) + self.pos_emb

        t_embed = self.flow_timestep_encoder(self._to_batch_vector(timestep, B, sample))
        tt_embed = self.flow_target_t_encoder(self._to_batch_vector(target_t, B, sample))
        time_c = self.timestep_target_t_adaptor(torch.cat([t_embed, tt_embed], dim=-1))

        context_c = self.vis_cond_obs_emb(global_cond) + self.vis_cond_pos_embed[:, :global_cond.shape[1]]

        for block in self.blocks:
            x = block(x, time_c, context_c)

        return self.final_layer(x)[:, -self.horizon:]

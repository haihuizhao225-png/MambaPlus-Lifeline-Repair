# -*- coding: utf-8 -*-
"""MambaPlus aligned with the original Chapter 3 mathematical Eqs. (1)-(9).

The Mamba backbone below implements the selective SSM recurrence with pure
PyTorch, not the fused CUDA mamba-ssm kernel. Each *feature* is one sequence
step; text token sequences have already been pooled by frozen BERT.

Fig. 1 post-modules: cross-level skip fusion, dynamic LayerNorm, FFN, residual fusion.

Full-factorial configurations (M1, M2, M3 are independently enabled):
  mamba, m1, m2, m3, m1_m2, m1_m3, m2_m3, mambaplus.
If an earlier module is absent, the next enabled module consumes the latest
available feature-token representation. This fallback is an explicit ablation
implementation choice and should be stated in methods.
"""

import math
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


class FeatureTokenizer(nn.Module):
    """Turn numerical/categorical columns and pooled BERT into feature tokens."""

    def __init__(
        self,
        n_numerical: int,
        categorical_cardinalities: List[int],
        with_text: bool,
        d_model: int,
        dropout: float,
    ):
        super().__init__()
        self.n_numerical = n_numerical
        self.n_categorical = len(categorical_cardinalities)
        self.with_text = with_text
        self.n_tokens = n_numerical + self.n_categorical + int(with_text)
        if self.n_tokens < 1:
            raise ValueError('At least one feature token is needed')

        if n_numerical:
            self.numeric_weight = nn.Parameter(torch.empty(n_numerical, d_model))
            self.numeric_bias = nn.Parameter(torch.zeros(n_numerical, d_model))
            nn.init.normal_(self.numeric_weight, std=0.02)

        self.category_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, d_model, padding_idx=0)
            for cardinality in categorical_cardinalities
        ])
        self.text_projection = (
            nn.Sequential(nn.LayerNorm(768), nn.Linear(768, d_model), nn.GELU())
            if with_text else None
        )
        self.field_bias = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        nn.init.normal_(self.field_bias, std=0.01)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        numerical: Tensor,
        categorical: Tensor,
        bert_vector: Optional[Tensor],
    ) -> Tensor:
        tokens = []
        if self.n_numerical:
            # [B, N_num] -> [B, N_num, D]
            tokens.append(
                numerical.unsqueeze(-1) * self.numeric_weight.unsqueeze(0)
                + self.numeric_bias.unsqueeze(0)
            )
        for i, embedding in enumerate(self.category_embeddings):
            tokens.append(embedding(categorical[:, i]).unsqueeze(1))
        if self.with_text:
            if bert_vector is None or bert_vector.shape[-1] != 768:
                raise ValueError('Expected frozen BERT pooled vectors of shape [B, 768]')
            tokens.append(self.text_projection(bert_vector).unsqueeze(1))
        x = torch.cat(tokens, dim=1)
        return self.dropout(self.norm(x + self.field_bias))


class SelectiveSSMMixer(nn.Module):
    """Reference selective state-space update, scanning L feature positions.

    This explicitly uses input-dependent delta, B and C, unlike the legacy
    one-step approximation. Runtime is O(L), but the Python loop is slower
    than a specialized fused selective-scan kernel.
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, d_conv: int = 4):
        super().__init__()
        inner = expand * d_model
        rank = max(1, math.ceil(d_model / 16))
        self.inner = inner
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, inner * 2, bias=False)
        self.conv = nn.Conv1d(inner, inner, d_conv, groups=inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(inner, rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(rank, inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).expand(inner, -1).clone())
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, d_model, bias=False)
        # Initialize positive step sizes in the typical small-value range.
        with torch.no_grad():
            dt = torch.exp(torch.rand(inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
            inv_softplus = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.copy_(inv_softplus)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[:, :, :length].transpose(1, 2)
        u = F.silu(u)
        params = self.x_proj(u)
        dt_low, b_values, c_values = torch.split(
            params, [self.dt_proj.in_features, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_low))
        a = -torch.exp(self.A_log.float()).to(dtype=u.dtype)
        state = u.new_zeros(batch, self.inner, self.d_state)
        ys = []
        for t in range(length):
            delta = dt[:, t, :].unsqueeze(-1)  # [B, E, 1]
            u_t = u[:, t, :]
            d_a = torch.exp(delta * a.unsqueeze(0))
            d_b_u = delta * b_values[:, t, :].unsqueeze(1) * u_t.unsqueeze(-1)
            state = d_a * state + d_b_u
            y_t = (state * c_values[:, t, :].unsqueeze(1)).sum(dim=-1) + self.D * u_t
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        return self.out_proj(y * F.silu(z))


class MambaResidualBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, expand: int, d_conv: int, dropout: float):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = SelectiveSSMMixer(d_model, d_state, expand, d_conv)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.dropout(self.mixer(self.norm(x)))


class ScaleResidualMLP(nn.Module):
    """A scale-specific residual MLP: D -> width -> residual MLP -> D.

    The residual connection occurs inside each branch, as shown in manuscript
    Fig. 2, in addition to the learnable *outer* residual in Eq. (1).
    Every operation is position-wise on [batch, feature_tokens, d_model].
    """

    def __init__(self, d_model: int, width: int, dropout: float):
        super().__init__()
        self.down = nn.Linear(d_model, width)
        self.residual_mlp = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
            nn.Dropout(dropout),
        )
        self.up = nn.Linear(width, d_model)

    def forward(self, x: Tensor) -> Tensor:
        z = self.down(x)
        return self.up(z + self.residual_mlp(z))


class HierarchicalFeatureModule(nn.Module):
    """M1 — original Chapter 3 Eq. (1), in feature-token space.

    F_M1 = LN(F_m + alpha_pyr * sum_s lambda_s * H_s(X))
    lambda_s = softmax(scale_logits)_s; H_s is a residual MLP branch of
    width D, D/2 or D/4 with its output restored to D dimensions.
    X and F_m both have shape [B, L, D].
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        dims = [d_model, max(8, d_model // 2), max(8, d_model // 4)]
        self.scales = nn.ModuleList([
            ScaleResidualMLP(d_model, width, dropout) for width in dims
        ])
        self.scale_logits = nn.Parameter(torch.zeros(3))
        self.residual_scale = nn.Parameter(torch.tensor(0.1))  # alpha_pyr
        self.norm = nn.LayerNorm(d_model)

    def forward(self, projected: Tensor, backbone: Tensor) -> Tensor:
        if projected.shape != backbone.shape or projected.ndim != 3:
            raise ValueError('M1 expects X and F_m of shape [B, L, D]')
        weights = torch.softmax(self.scale_logits, dim=0)
        multiscale = sum(
            weight * branch(projected)
            for weight, branch in zip(weights, self.scales)
        )
        return self.norm(backbone + self.residual_scale * multiscale)


class AdaptiveGatingModule(nn.Module):
    """M2 — original Chapter 3 Eqs. (2)-(4), exactly.

    g_gate = sigmoid(MLP_gate(F_M1))
    F_gate = g_gate elementwise-multiplied by F_M1
    F_M2 = LN(F_m + alpha_gate * F_gate)

    Gate weights are token/channel weights, not automatically the importance
    of raw engineering columns. All tensors have shape [B, L, D].
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))  # alpha_gate
        self.norm = nn.LayerNorm(d_model)

    def forward(self, m1: Tensor, backbone: Tensor) -> Tensor:
        if m1.shape != backbone.shape or m1.ndim != 3:
            raise ValueError('M2 expects F_M1 and F_m of shape [B, L, D]')
        gate_weights = self.gate(m1)
        gated_feature = gate_weights * m1
        return self.norm(backbone + self.residual_scale * gated_feature)


class MultiLevelAttentionModule(nn.Module):
    """M3 — original Chapter 3 Eqs. (5)-(9), in feature-token space.

    X_F   = MHA(X)
    X_C   = X * sigmoid(MLP_C(GAP_L(X)))
    R_L   = softmax(Q K^T / sqrt(D) + learnable field-relation bias)
    X_L   = X * sigmoid(MLP_L(Pool(R_L)))
    X_att = Fusion(X_F, X_C, X_L)
    F_3   = LN(F_M2 + alpha_att * g_att * X_att)

    Unlike the earlier draft implementation, all three attention branches
    return [B, L, D] and the prior Module-2 token representation is used as X.
    Final pooling to [B, D] is performed only after M3, in the regressor.
    """

    def __init__(self, d_model: int, n_tokens: int, dropout: float, heads: int = 4):
        super().__init__()
        if d_model % heads:
            raise ValueError('d_model must be divisible by attention heads')
        if n_tokens < 1:
            raise ValueError('M3 requires at least one feature token')
        self.n_tokens = n_tokens
        self.feature_attention = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        channel_width = max(8, d_model // 4)
        self.channel_gate = nn.Sequential(
            nn.Linear(d_model, channel_width),
            nn.GELU(),
            nn.Linear(channel_width, d_model),
            nn.Sigmoid(),
        )
        self.field_q = nn.Linear(d_model, d_model, bias=False)
        self.field_k = nn.Linear(d_model, d_model, bias=False)
        # Unlike standard attention value aggregation, paper Eq. (7) gates X
        # using a pooled FIELD-RELATION matrix. No field_v is needed.
        self.field_relation_bias = nn.Parameter(torch.zeros(n_tokens, n_tokens))
        self.field_gate = nn.Sequential(
            nn.Linear(n_tokens, max(8, 2 * n_tokens)),
            nn.GELU(),
            nn.Linear(max(8, 2 * n_tokens), n_tokens),
            nn.Sigmoid(),
        )
        self.fusion_weights = nn.Parameter(torch.zeros(3))
        self.residual_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))  # alpha_att
        self.norm = nn.LayerNorm(d_model)

    def forward(self, module2_tokens: Tensor) -> Tensor:
        if module2_tokens.ndim != 3 or module2_tokens.size(1) != self.n_tokens:
            raise ValueError('M3 expects F_M2 tokens of shape [B, n_tokens, D]')
        x = module2_tokens

        # Eq. (5): feature-token multi-head attention, retaining all fields.
        x_feature, _ = self.feature_attention(x, x, x, need_weights=False)

        # Eq. (6): channel attention; GAP across the field/sequence axis.
        channel_weights = self.channel_gate(x.mean(dim=1)).unsqueeze(1)
        x_channel = x * channel_weights

        # Eq. (7): field-level positional relations, followed by a field gate.
        # Pool over query positions -> target-field incoming relation strength.
        # A mean over the LAST axis after row-softmax would be identically 1/L;
        # pooling over queries avoids this degenerate constant.
        q, k = self.field_q(x), self.field_k(x)
        relation_logits = q @ k.transpose(1, 2) / math.sqrt(q.size(-1))
        relation_logits = relation_logits + self.field_relation_bias.unsqueeze(0)
        relations = torch.softmax(relation_logits, dim=-1)  # [B, L, L]
        field_summary = relations.mean(dim=1)  # [B, L] incoming mass
        field_weights = self.field_gate(field_summary).unsqueeze(-1)
        x_field = x * field_weights

        # Eq. (8): complementary fusion, preserving field/channel alignment.
        w = torch.softmax(self.fusion_weights, dim=0)
        x_attention = w[0] * x_feature + w[1] * x_channel + w[2] * x_field

        # Eq. (9): residual gate operates element-wise over feature tokens.
        attention_gate = self.residual_gate(torch.cat([x, x_attention], dim=-1))
        return self.norm(x + self.residual_scale * attention_gate * x_attention)


class CrossLevelSkipFusion(nn.Module):
    """Fig. 1: skip aggregation across the backbone and activated M1/M2/M3 stages.

    These tensors all have shape [B, L, D]. Stage parameters are fixed to
    four slots, so the subsequent post-fusion head is identical in all
    progressive ablations. Only present stages participate in softmax.
    """

    STAGES = ('backbone', 'm1', 'm2', 'm3')

    def __init__(self):
        super().__init__()
        self.stage_logits = nn.Parameter(torch.zeros(len(self.STAGES)))

    def forward(self, stages: Dict[str, Tensor]) -> Tensor:
        if not stages or 'backbone' not in stages:
            raise ValueError('Cross-level fusion requires a backbone representation')
        keys = [key for key in self.STAGES if key in stages]
        base_shape = stages['backbone'].shape
        if any(stages[key].shape != base_shape for key in keys):
            raise ValueError('All skip levels must have identical [B,L,D] shapes')
        positions = [self.STAGES.index(key) for key in keys]
        weights = torch.softmax(self.stage_logits[positions], dim=0)
        return sum(weight * stages[key] for weight, key in zip(weights, keys))


class DynamicLayerNorm(nn.Module):
    """Fig. 1: conditional LN after cross-level skip aggregation.

    LN is modulated by sample-dependent scale/shift inferred from the global
    feature-token context. The last projection is initialized to zero, so
    the initial behavior is standard LayerNorm.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        bottleneck = max(8, d_model // 4)
        self.condition = nn.Sequential(
            nn.Linear(d_model, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 2 * d_model),
        )
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError('DynamicLayerNorm requires [B,L,D]')
        delta_scale, delta_shift = self.condition(x.mean(dim=1)).chunk(2, dim=-1)
        gamma = 1.0 + 0.1 * torch.tanh(delta_scale).unsqueeze(1)
        beta = 0.1 * torch.tanh(delta_shift).unsqueeze(1)
        return self.norm(x) * gamma + beta


class PostModuleRefinement(nn.Module):
    """Fig. 1: Cross-level skips -> Dynamic LayerNorm -> FFN -> residual fusion."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.cross_level = CrossLevelSkipFusion()
        self.dynamic_norm = DynamicLayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.residual_norm = nn.LayerNorm(d_model)

    def forward(self, levels: Dict[str, Tensor]) -> Tensor:
        fused = self.cross_level(levels)
        refined = self.ffn(self.dynamic_norm(fused))
        return self.residual_norm(fused + self.residual_scale * refined)


class MambaPlusRegressor(nn.Module):
    """MambaPlus and full 2^3 factorial M1/M2/M3 ablations.

    The complete configuration follows Chapter 3 Eqs. (1)-(9). For ablations,
    disabled stages are bypassed and downstream enabled stages consume the most
    recent representation, with the Mamba backbone as the shared residual.
    The Fig. 1 post-refinement remains common to all variants.
    """

    ABLATION_VARIANTS = {
        'mamba': (),
        'm1': ('m1',),
        'm2': ('m2',),
        'm3': ('m3',),
        'm1_m2': ('m1', 'm2'),
        'm1_m3': ('m1', 'm3'),
        'm2_m3': ('m2', 'm3'),
        'mambaplus': ('m1', 'm2', 'm3'),
    }

    def __init__(
        self,
        n_numerical: int,
        categorical_cardinalities: List[int],
        with_text: bool,
        variant: str = 'mambaplus',
        d_model: int = 384,
        n_layers: int = 8,
        dropout: float = 0.025,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        feature_order: str = 'normal',
        order_seed: int = 2026,
    ):
        super().__init__()
        if variant not in self.ABLATION_VARIANTS:
            raise ValueError(f'Unknown variant: {variant}; choose from {list(self.ABLATION_VARIANTS)}')
        self.variant = variant
        self.enabled_modules = self.ABLATION_VARIANTS[variant]
        self.tokenizer = FeatureTokenizer(
            n_numerical, categorical_cardinalities, with_text, d_model, dropout
        )
        if feature_order == 'normal':
            ordering = torch.arange(self.tokenizer.n_tokens)
        elif feature_order == 'reverse':
            ordering = torch.arange(self.tokenizer.n_tokens - 1, -1, -1)
        elif feature_order == 'shuffled':
            rng = torch.Generator().manual_seed(order_seed)
            ordering = torch.randperm(self.tokenizer.n_tokens, generator=rng)
        else:
            raise ValueError('feature_order must be normal, reverse or shuffled')
        self.register_buffer('token_permutation', ordering)
        self.backbone = nn.ModuleList([
            MambaResidualBlock(d_model, d_state, expand, d_conv, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        if 'm1' in self.enabled_modules:
            self.m1 = HierarchicalFeatureModule(d_model, dropout)
        if 'm2' in self.enabled_modules:
            self.m2 = AdaptiveGatingModule(d_model, dropout)
        if 'm3' in self.enabled_modules:
            self.m3 = MultiLevelAttentionModule(d_model, self.tokenizer.n_tokens, dropout)
        self.post_refinement = PostModuleRefinement(d_model, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, numerical: Tensor, categorical: Tensor,
                bert_vector: Optional[Tensor] = None) -> Tensor:
        x = self.tokenizer(numerical, categorical, bert_vector)
        x = x.index_select(1, self.token_permutation)
        fm = x
        for block in self.backbone:
            fm = block(fm)
        fm = self.final_norm(fm)
        levels = {'backbone': fm}
        current = fm
        if 'm1' in self.enabled_modules:
            current = self.m1(x, fm)
            levels['m1'] = current
        if 'm2' in self.enabled_modules:
            current = self.m2(current, fm)
            levels['m2'] = current
        if 'm3' in self.enabled_modules:
            current = self.m3(current)
            levels['m3'] = current
        tokens = self.post_refinement(levels)
        return self.head(tokens.mean(dim=1)).squeeze(-1)

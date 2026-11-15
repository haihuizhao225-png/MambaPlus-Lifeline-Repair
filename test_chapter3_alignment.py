# -*- coding: utf-8 -*-
"""Deterministic structural and equation tests for Chapter 3 Eqs. (1)-(9).

No pretrained BERT downloads or research dataset are needed.
Run: python test_chapter3_alignment.py
"""
import math

import torch

from model import (
    AdaptiveGatingModule, HierarchicalFeatureModule,
    MambaPlusRegressor, MultiLevelAttentionModule,
)


def require_close(name, actual, expected, atol=1e-6):
    if not torch.allclose(actual, expected, atol=atol, rtol=1e-5):
        err = (actual - expected).abs().max().item()
        raise AssertionError(f'{name}: mismatch (max abs error = {err})')


def check_equation_1():
    torch.manual_seed(11)
    b, l, d = 3, 10, 32
    m1 = HierarchicalFeatureModule(d, dropout=0.0).eval()
    x, fm = torch.randn(b, l, d), torch.randn(b, l, d)
    weights = torch.softmax(m1.scale_logits, dim=0)
    expected = m1.norm(fm + m1.residual_scale * sum(
        w * branch(x) for w, branch in zip(weights, m1.scales)
    ))
    actual = m1(x, fm)
    assert actual.shape == (b, l, d)
    require_close('Eq. (1)', actual, expected)
    for branch in m1.scales:
        # Fig. 2: each branch genuinely contains an internal residual MLP.
        z = branch.down(x)
        require_close('M1 residual branch', branch(x), branch.up(z + branch.residual_mlp(z)))


def check_equations_2_to_4():
    torch.manual_seed(12)
    b, l, d = 3, 10, 32
    m2 = AdaptiveGatingModule(d, dropout=0.0).eval()
    fm1, fm = torch.randn(b, l, d), torch.randn(b, l, d)
    gate = m2.gate(fm1)  # Eq. (2)
    assert bool(torch.all((gate >= 0) & (gate <= 1)))
    f_gate = gate * fm1  # Eq. (3)
    expected = m2.norm(fm + m2.residual_scale * f_gate)  # Eq. (4)
    require_close('Eqs. (2)-(4)', m2(fm1, fm), expected)


def check_equations_5_to_9():
    torch.manual_seed(13)
    b, l, d = 3, 10, 32
    m3 = MultiLevelAttentionModule(d, n_tokens=l, dropout=0.0, heads=4).eval()
    x = torch.randn(b, l, d)
    x_feature, _ = m3.feature_attention(x, x, x, need_weights=False)  # Eq. (5)
    x_channel = x * m3.channel_gate(x.mean(dim=1)).unsqueeze(1)  # Eq. (6)
    q, k = m3.field_q(x), m3.field_k(x)
    scores = q @ k.transpose(1, 2) / math.sqrt(d) + m3.field_relation_bias.unsqueeze(0)
    relations = torch.softmax(scores, dim=-1)
    # Eq. (7): Pool over queries; field scores remain input dependent.
    x_field = x * m3.field_gate(relations.mean(dim=1)).unsqueeze(-1)
    weights = torch.softmax(m3.fusion_weights, dim=0)
    x_att = weights[0] * x_feature + weights[1] * x_channel + weights[2] * x_field  # (8)
    residual_gate = m3.residual_gate(torch.cat([x, x_att], dim=-1))
    expected = m3.norm(x + m3.residual_scale * residual_gate * x_att)  # Eq. (9)
    require_close('Eqs. (5)-(9)', m3(x), expected)
    assert expected.shape == (b, l, d)
    assert not torch.allclose(relations.mean(dim=1), torch.full((b, l), 1.0 / l))


def check_all_ablation_variants_and_gradients():
    for n_numeric, cat_cardinalities, with_text in [
        (6, [4, 4, 4], True),   # full 10 features
        (6, [4, 4, 4], False),  # no text, 9 features
        (5, [4, 4, 4], True),   # no RepairSize, 9 features
        (5, [4, 4, 4], False),  # no size or text, 8 features
        (6, [4, 4], True),      # LOCO, 9 features
        (5, [4, 4], False),     # LOCO reduced, 7 features
    ]:
        for variant in list(MambaPlusRegressor.ABLATION_VARIANTS):
            torch.manual_seed(25)
            net = MambaPlusRegressor(
                n_numerical=n_numeric, categorical_cardinalities=cat_cardinalities,
                with_text=with_text, variant=variant, d_model=32, n_layers=1,
                dropout=0.0, d_state=4, feature_order='reverse'
            )
            net.train()
            num = torch.randn(3, n_numeric)
            cat = torch.ones(3, len(cat_cardinalities), dtype=torch.long)
            bert = torch.randn(3, 768) if with_text else None
            y = net(num, cat, bert)
            assert y.shape == (3,), (variant, y.shape)
            y.square().mean().backward()
            assert net.backbone[0].mixer.dt_proj.weight.grad is not None
            if 'm1' in net.enabled_modules:
                assert net.m1.scales[0].residual_mlp[1].weight.grad is not None
            if 'm2' in net.enabled_modules:
                assert net.m2.gate[0].weight.grad is not None
            if 'm3' in net.enabled_modules:
                assert net.m3.field_gate[0].weight.grad is not None
                assert net.m3.field_relation_bias.grad is not None
                assert net.m3.residual_gate[0].weight.grad is not None


if __name__ == '__main__':
    torch.set_num_threads(2)
    check_equation_1()
    check_equations_2_to_4()
    check_equations_5_to_9()
    check_all_ablation_variants_and_gradients()
    print('PASS: Chapter 3 equations (1)-(9), branch residuals, all ablations, gradients.')


def check_figure1_post_modules():
    """Paper Fig.1: explicit cross-level skip, conditional LayerNorm, FFN and residual."""
    from model import CrossLevelSkipFusion, DynamicLayerNorm, PostModuleRefinement
    torch.manual_seed(44)
    b, l, d = 3, 10, 32
    levels = {k: torch.randn(b, l, d, requires_grad=True)
              for k in ('backbone', 'm1', 'm2', 'm3')}
    post = PostModuleRefinement(d, dropout=0.0).eval()
    fused = post.cross_level(levels)
    weights = torch.softmax(post.cross_level.stage_logits, dim=0)
    expected_fused = sum(w * levels[k] for w, k in zip(weights, post.cross_level.STAGES))
    require_close('Cross-level skip fusion', fused, expected_fused)
    dynamic = post.dynamic_norm(fused)
    refined = post.ffn(dynamic)
    expected_out = post.residual_norm(fused + post.residual_scale * refined)
    require_close('Post-module residual fusion', post(levels), expected_out)
    assert expected_out.shape == (b, l, d)
    expected_out.square().mean().backward()
    assert post.cross_level.stage_logits.grad is not None
    assert post.dynamic_norm.condition[-1].weight.grad is not None
    assert post.ffn[0].weight.grad is not None
    assert levels['m3'].grad is not None
    # At initialization, dynamic LN reduces to ordinary LN.
    norm = DynamicLayerNorm(d).eval()
    require_close('Dynamic LN init', norm(fused.detach()), norm.norm(fused.detach()))
    # When only the backbone exists, the same post-refinement remains valid.
    require_close('Single-level fusion', post.cross_level({'backbone': levels['backbone']}), levels['backbone'])


if __name__ == '__main__':
    check_figure1_post_modules()
    print('PASS: Fig.1 cross-level skip, Dynamic LayerNorm, FFN and residual fusion.')

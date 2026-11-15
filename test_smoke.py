# -*- coding: utf-8 -*-
"""Tests do NOT require downloading BERT; they test the downstream model and splits."""
import numpy as np
import pandas as pd
import torch
from model import MambaPlusRegressor
from train import (
    TrainOnlyPreprocessor, make_splits, RepairDataset, EXPERIMENTS,
    feature_columns,
)


def fake_dataframe(n=120):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'RepairSize': rng.uniform(0.5, 5, size=n),
        'Sa': rng.uniform(0.2, 1.0, size=n),
        'PGA': rng.uniform(0.1, 1.5, size=n),
        'Mw': rng.choice([6.2, 7.1], size=n),
        'ZZJ': np.linspace(0.5, 50, n),
        'ZYJ': np.linspace(5, 60, n),
        'System': np.repeat(['Sewerage', 'Water Supply', 'Storm Water'], n // 3),
        'Element': rng.choice(['Pipe', 'Joint', 'Manhole'], size=n),
        'ElementMaterial': rng.choice(['PVC', 'Concrete', 'Steel'], size=n),
        'ElementDamage': rng.choice(['Minor', 'Moderate'], size=n),
        'RepairDescription': rng.choice(['Replace cracked pipe', 'Repair damaged joint'], size=n),
        'RepairTime': rng.uniform(0.1, 80.0, size=n),
        '_source_row': np.arange(n),
    })


def test_variants_backward():
    for variant in list(MambaPlusRegressor.ABLATION_VARIANTS):
        net = MambaPlusRegressor(
            n_numerical=6, categorical_cardinalities=[4, 4, 4], with_text=True,
            variant=variant, d_model=32, n_layers=1,
        )
        out = net(torch.randn(3, 6), torch.tensor([[1, 2, 3], [2, 0, 1], [1, 2, 2]]), torch.randn(3, 768))
        assert out.shape == (3,)
        out.square().mean().backward()
        assert net.backbone[0].mixer.dt_proj.weight.grad is not None
        assert net.tokenizer.n_tokens == 10


def test_train_only_scaler_unknown_labels():
    df = fake_dataframe()
    train, val, test = make_splits(df, 'random_no_text', 'Element', 42)[0][1:]
    numeric, categoric, text = feature_columns(df, { 'RepairDescription' })
    pre = TrainOnlyPreprocessor(numeric, categoric).fit(df.iloc[train])
    assert np.isclose(pre.target_scaler.mean_[0], df.iloc[train]['RepairTime'].mean())
    assert not np.isclose(pre.target_scaler.mean_[0], df['RepairTime'].mean())
    new = df.iloc[test].copy()
    new['ElementMaterial'] = 'unseen-new-material'
    _, encoded, _ = pre.transform(new)
    assert np.all(encoded[:, categoric.index('ElementMaterial')] == 0)


def test_loco_disjoint_and_no_element():
    df = fake_dataframe()
    folds = make_splits(df, 'loco', 'System', 42)
    assert len(folds) == 3
    for name, train, val, test in folds:
        assert not (set(df.iloc[test].System) & set(df.iloc[train].System))
        assert len(set(train) | set(val) | set(test)) == len(df)
    nums, cats, has_text = feature_columns(df, EXPERIMENTS['loco'][1])
    assert 'Element' in cats and has_text  # System is NOT one of the model inputs.
    assert len(nums) + len(cats) + int(has_text) == 10


def test_distance_bidirectional():
    df = fake_dataframe()
    folds = make_splits(df, 'distance_zzj', 'Element', 42)
    assert len(folds) == 2
    for _, tr, va, te in folds:
        assert not set(tr) & set(te)
        assert not set(va) & set(te)
        assert len(set(tr) | set(va) | set(te)) == len(df)


if __name__ == '__main__':
    test_variants_backward()
    test_train_only_scaler_unknown_labels()
    test_loco_disjoint_and_no_element()
    test_distance_bidirectional()
    print('All smoke tests passed.')

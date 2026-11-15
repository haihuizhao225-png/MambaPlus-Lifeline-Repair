# -*- coding: utf-8 -*-
"""Pipeline-only smoke test; uses a FAKE estimator, NEVER research results."""
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_mamba_models import run_single, metrics_on_original_scale
from train import read_data, make_splits


class FakeRegressor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y, *, X_val, y_val, embeddings=None, embeddings_val=None, **kw):
        assert X.shape[0] == len(y) and X_val.shape[0] == len(y_val)
        if embeddings is not None:
            assert len(embeddings) == 1 and embeddings[0].shape[1] == 768
            assert embeddings_val[0].shape[1] == 768
        self.reference = float(np.mean(y))

    def predict(self, X, embeddings=None):
        if embeddings is not None:
            assert embeddings[0].shape[1] == 768
        return np.full(len(X), self.reference, dtype=np.float32)


def run():
    df = read_data(str(Path(__file__).with_name('example_synthetic_data.csv')))
    split = make_splits(df, 'random_full', 'Element', 42)[0]
    with tempfile.TemporaryDirectory() as temp:
        args = SimpleNamespace(output=temp, d_model=16, n_layers=1, epochs=1,
                               batch_size=12, patience=1, lr=.001, weight_decay=0)
        fake_bert = np.arange(len(df) * 768, dtype=np.float32).reshape(len(df), 768)
        with patch('baseline_mamba_models.get_model_class', return_value=FakeRegressor), \
             patch('importlib.metadata.version', return_value='FAKE-TEST-ONLY'):
            a = run_single(df, fake_bert, split, 'random_full', 'mambular', 42, args)
            assert a['with_frozen_bert'] and a['n_train'] + a['n_val'] + a['n_test'] == len(df)
            b = run_single(df, None, split, 'random_no_text', 'mambatab', 42, args)
            assert not b['with_frozen_bert']
            assert (Path(temp) / 'random_full' / 'mambular' / 'random_70_10_20' / 'seed_42' / 'test_predictions.csv').exists()
            assert (Path(temp) / 'random_no_text' / 'mambatab' / 'random_70_10_20' / 'seed_42' / 'labor_hour_strata.csv').exists()
    assert metrics_on_original_scale(np.array([1., 2., 3.]), np.array([1., 2., 3.]))['r2'] == 1.
    print('PASS: splitting / preprocessing / embeddings / reporting pipeline (fake estimator).')
    print('NOTE: Official mambular package and GPU training were NOT tested here.')


if __name__ == '__main__':
    run()

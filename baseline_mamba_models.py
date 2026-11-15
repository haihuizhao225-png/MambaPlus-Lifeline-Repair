# -*- coding: utf-8 -*-
"""MambaTab + Mambular regression benchmark for the revised MambaPlus paper.

Uses the same data reading, fixed random split, train-only preprocessing and
frozen BERT vectors as train.py. Supports a real official mambular package.
Does not fabricate results or silently fall back to an alternative architecture.

Example:
  python baseline_mamba_models.py --data "F:\\data\\lifeline.csv" \
      --models mambatab,mambular --seeds 42,52,62,72,82
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train import (
    EXPERIMENTS, TARGET, TEXT, TrainOnlyPreprocessor,
    compute_frozen_bert_embeddings, feature_columns, fix_seed,
    make_splits, read_data,
)

FEATURE_EXPERIMENTS = {
    'random_full', 'random_no_text', 'random_no_size', 'random_no_both'
}


def get_model_class(name: str):
    try:
        from mambular.models import MambaTabRegressor, MambularRegressor
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            'Missing official mambular models. Install with: pip install mambular. '
            'The upstream MambaTab repository primarily offers classification; '
            'we intentionally use its regression implementation in mambular.'
        ) from exc
    return {'mambatab': MambaTabRegressor, 'mambular': MambularRegressor}[name]


def select_facilities(df: pd.DataFrame, component_col: str, values: str | None) -> pd.DataFrame:
    if values is None:
        if component_col in df and df[component_col].nunique(dropna=True) > 3:
            print('[WARNING] Facility filter missing; dataset may include other lifeline systems. '
                  'Supply --component-col System --facility-values "Sewerage,Water Supply,Storm Water" if needed.')
        return df
    expected = [part.strip() for part in values.split(',') if part.strip()]
    if len(expected) != 3:
        raise ValueError('Exactly three facility names are required for --facility-values')
    if component_col not in df:
        raise KeyError(f'Missing facility grouping column: {component_col}')
    actual = df[component_col].fillna('').astype(str).str.strip()
    mask = actual.str.casefold().isin(set(v.casefold() for v in expected))
    result = df.loc[mask].reset_index(drop=True)
    observed = set(result[component_col].astype(str).str.strip().str.casefold())
    if observed != set(v.casefold() for v in expected):
        raise ValueError(f'Facility categories not found. Selected: {result[component_col].value_counts().to_dict()}')
    print(f'[DATA] selected facility counts: {result[component_col].value_counts().to_dict()}')
    return result


def build_inputs(num: np.ndarray, cat: np.ndarray, nums: List[str], cats: List[str]) -> pd.DataFrame:
    """Keep categorical tokens categorical (not misdetected as continuous)."""
    cols: Dict[str, object] = {}
    for i, name in enumerate(nums):
        cols[name] = num[:, i].astype('float32')
    for i, name in enumerate(cats):
        cols[name] = np.asarray([f'category_{v}' for v in cat[:, i]], dtype=object)
    return pd.DataFrame(cols)


def metrics_on_original_scale(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size != y_pred.size or not np.isfinite(y_pred).all():
        raise ValueError('Prediction size mismatch or nonfinite predictions')
    return {
        'r2': float(r2_score(y_true, y_pred)),
        'mae_person_h': float(mean_absolute_error(y_true, y_pred)),
        'rmse_person_h': float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def run_single(df: pd.DataFrame, bert: np.ndarray | None, split,
               experiment: str, model_name: str, seed: int, args) -> dict:
    fold, train_idx, val_idx, test_idx = split
    if any(np.intersect1d(a, b).size for a, b in
           ((train_idx, val_idx), (train_idx, test_idx), (val_idx, test_idx))):
        raise AssertionError('Data leakage: overlapping train/val/test indices')
    _, omitted = EXPERIMENTS[experiment]
    nums, cats, with_text = feature_columns(df, omitted)
    if with_text and bert is None:
        raise RuntimeError('Frozen BERT vectors are missing for text-enabled benchmark')

    train_df, val_df, test_df = (df.iloc[idx] for idx in (train_idx, val_idx, test_idx))
    pp = TrainOnlyPreprocessor(nums, cats).fit(train_df)
    a_num, a_cat, y_train = pp.transform(train_df)
    v_num, v_cat, y_val = pp.transform(val_df)
    t_num, t_cat, y_test = pp.transform(test_df)
    X_train, X_val, X_test = (
        build_inputs(a_num, a_cat, nums, cats),
        build_inputs(v_num, v_cat, nums, cats),
        build_inputs(t_num, t_cat, nums, cats),
    )
    # Pretrained frozen BERT: SINGLE 768-dimensional semantic embedding per text field.
    # Crucially, DO NOT flatten 768 coordinates into 768 Mamba feature tokens.
    train_embeddings = [bert[train_idx].astype('float32')] if with_text else None
    val_embeddings = [bert[val_idx].astype('float32')] if with_text else None
    test_embeddings = [bert[test_idx].astype('float32')] if with_text else None
    if set(X_train.columns) != set(X_val.columns) or list(X_train.columns) != list(X_test.columns):
        raise AssertionError('Misaligned feature columns')

    fix_seed(seed)
    clazz = get_model_class(model_name)
    settings = {'d_model': args.d_model, 'n_layers': args.n_layers,
                'numerical_preprocessing': 'standardization',
                'categorical_preprocessing': 'int'}
    if model_name == 'mambular':
        # Pure-PyTorch Mamba backend avoids requiring mamba-ssm CUDA extensions.
        settings['mamba_version'] = 'mamba-torch'
    model = clazz(**settings)
    out = Path(args.output) / experiment / model_name / fold / f'seed_{seed}'
    out.mkdir(parents=True, exist_ok=True)
    print(f'\n[RUN] model={model_name}, experiment={experiment}, seed={seed}, '
          f'train/val/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)}, text={with_text}')
    started = time.perf_counter()
    fit_options = dict(
        X_val=X_val, y_val=y_val,
        max_epochs=args.epochs,
        random_state=seed,
        batch_size=args.batch_size,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_path=str(out / 'checkpoints'),
    )
    if with_text:
        fit_options['embeddings'] = train_embeddings
        fit_options['embeddings_val'] = val_embeddings
    try:
        model.fit(X_train, y_train, **fit_options)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f'{model_name} failed with official Mambular API. Check installed package version '
            'and support for external semantic embeddings. The script deliberately '
            'does NOT drop the text embedding or substitute a different model.'
        ) from exc
    train_seconds = time.perf_counter() - started
    pred_scaled = model.predict(X_test, embeddings=test_embeddings) if with_text else model.predict(X_test)
    pred_scaled = np.asarray(pred_scaled).reshape(-1)
    actual = pp.inverse_target(y_test)
    pred = pp.inverse_target(pred_scaled)
    scores = metrics_on_original_scale(actual, pred)

    predictions = pd.DataFrame({
        'source_row': test_df['_source_row'].to_numpy(),
        'true_person_h': actual, 'pred_person_h': pred,
        'absolute_error_person_h': np.abs(actual - pred),
    })
    predictions.to_csv(out / 'test_predictions.csv', index=False, encoding='utf-8-sig')
    strata = []
    intervals = pd.cut(actual, [-np.inf, 8, 24, 100, np.inf], labels=['<=8', '8-24', '24-100', '>100'])
    for label in intervals.categories:
        m = np.asarray(intervals == label)
        if m.sum():
            sub = metrics_on_original_scale(actual[m], pred[m]) if m.sum() >= 2 else {
                'r2': np.nan, 'mae_person_h': float(abs(actual[m][0] - pred[m][0])),
                'rmse_person_h': float(abs(actual[m][0] - pred[m][0]))}
            strata.append({'interval_person_h': str(label), 'count': int(m.sum()), **sub})
    pd.DataFrame(strata).to_csv(out / 'labor_hour_strata.csv', index=False, encoding='utf-8-sig')
    record = dict(
        experiment=experiment, variant=model_name, fold=fold, train_seed=seed,
        n_train=len(train_idx), n_val=len(val_idx), n_test=len(test_idx),
        n_numerical=len(nums), n_categorical=len(cats),
        with_frozen_bert=with_text, text_embedding_dimension=768 if with_text else 0,
        training_seconds=train_seconds, **scores,
    )
    (out / 'run_settings.json').write_text(json.dumps({
        'run': record, 'model_settings': settings, 'fit_options': {
            k: v for k, v in fit_options.items() if k not in ('X_val', 'y_val', 'embeddings', 'embeddings_val')},
        'numeric_features': nums, 'categorical_features': cats,
        'target_scaler_mean': pp.target_scaler.mean_.tolist(),
        'target_scaler_scale': pp.target_scaler.scale_.tolist(),
        'mambular_version': __import__('importlib.metadata', fromlist=['version']).version('mambular'),
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[TEST]', record)
    return record


def parse_args():
    p = argparse.ArgumentParser(description='Official MambaTabRegressor and MambularRegressor benchmark')
    p.add_argument('--data', required=True, help='Original CSV / Excel, with RepairTime and raw RepairDescription')
    p.add_argument('--output', default='results_baselines')
    p.add_argument('--models', default='mambatab,mambular', help='mambatab,mambular')
    p.add_argument('--experiment', default='random_full', choices=sorted(FEATURE_EXPERIMENTS))
    p.add_argument('--component-col', default='System')
    p.add_argument('--facility-values', default=None)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--seeds', default='42,52,62,72,82')
    p.add_argument('--bert-model', default='bert-base-multilingual-cased')
    p.add_argument('--max-tokens', type=int, default=128)
    p.add_argument('--bert-batch', type=int, default=32)
    p.add_argument('--bert-cache', default='bert_embedding_cache')
    p.add_argument('--refresh-bert-cache', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--d-model', type=int, default=64, help='Baseline-specific width, not forced to MambaPlus=384')
    p.add_argument('--n-layers', type=int, default=4)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=96)
    p.add_argument('--lr', type=float, default=0.00015)
    p.add_argument('--weight-decay', type=float, default=1e-7)
    p.add_argument('--patience', type=int, default=80)
    args = p.parse_args()
    args.models = [s.strip().lower() for s in args.models.split(',') if s.strip()]
    if not args.models or set(args.models) - {'mambatab', 'mambular'}:
        p.error('--models must be mambatab,mambular or a subset')
    try:
        args.seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    except ValueError:
        p.error('--seeds must be a comma-separated integer list')
    if not args.seeds:
        p.error('At least one seed is required')
    return args


def main():
    args = parse_args()
    df = select_facilities(read_data(args.data), args.component_col, args.facility_values)
    _, excluded = EXPERIMENTS[args.experiment]
    feature_columns(df, excluded)
    if args.experiment == 'random_full' and len(df) != 14695:
        print(f'[NOTE] N={len(df):,}; verify that it matches the manuscript N=14,695.')
    bert = compute_frozen_bert_embeddings(df, args) if TEXT not in excluded else None
    splits = make_splits(df, args.experiment, args.component_col, args.split_seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for split in splits:
        for name in args.models:
            for seed in args.seeds:
                results.append(run_single(df, bert, split, args.experiment, name, seed, args))
                pd.DataFrame(results).to_csv(out / 'all_run_metrics.csv', index=False, encoding='utf-8-sig')
    table = pd.DataFrame(results)
    summary = table.groupby(['experiment', 'variant', 'fold'], dropna=False).agg(
        n_runs=('r2', 'size'),
        r2_mean=('r2', 'mean'), r2_sd=('r2', 'std'),
        mae_mean=('mae_person_h', 'mean'), mae_sd=('mae_person_h', 'std'),
        rmse_mean=('rmse_person_h', 'mean'), rmse_sd=('rmse_person_h', 'std'),
        training_seconds_mean=('training_seconds', 'mean'),
    ).reset_index()
    summary.to_csv(out / 'summary_mean_sd.csv', index=False, encoding='utf-8-sig')
    print('\n[FINISHED]\n', summary.to_string(index=False))
    print('[FILES]', out.resolve())


if __name__ == '__main__':
    main()

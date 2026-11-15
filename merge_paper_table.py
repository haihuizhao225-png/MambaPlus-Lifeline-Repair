# -*- coding: utf-8 -*-
"""Combine newly computed MambaPlus + MambaTab + Mambular runs into an SCI table.
Does NOT impute missing results or mix historical numbers.
"""
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mambaplus', default='results_mambaplus/all_run_metrics.csv')
    parser.add_argument('--baselines', default='results_baselines/all_run_metrics.csv')
    parser.add_argument('--output', default='results_paper_table.csv')
    args = parser.parse_args()
    frames = []
    for path in (args.mambaplus, args.baselines):
        p = Path(path)
        if not p.exists():
            print(f'[SKIP] file not found: {p}')
            continue
        df = pd.read_csv(p, encoding='utf-8-sig')
        missing = {'experiment', 'variant', 'fold', 'train_seed', 'r2', 'mae_person_h', 'rmse_person_h'} - set(df.columns)
        if missing:
            raise ValueError(f'{p}: missing columns {missing}')
        frames.append(df)
    if not frames:
        raise FileNotFoundError('No real metrics files found. First execute the training scripts.')
    raw = pd.concat(frames, ignore_index=True)
    # Only compare like-for-like test records and the ten-feature configuration.
    raw = raw[(raw.experiment == 'random_full') & (raw.fold == 'random_70_10_20')].copy()
    if 'feature_order' in raw.columns:
        raw = raw[raw.feature_order.isna() | raw.feature_order.eq('normal')]
    if raw.empty:
        raise ValueError('No comparable random_full / random_70_10_20 runs found')
    dup = raw.duplicated(subset=['variant', 'train_seed'], keep=False)
    if dup.any():
        raise ValueError(f'Duplicate model/seed records detected:\n{raw.loc[dup, ["variant", "train_seed"]]}')
    grouped = raw.groupby('variant', dropna=False).agg(
        N=('r2', 'size'), R2_mean=('r2', 'mean'), R2_sd=('r2', 'std'),
        MAE_mean=('mae_person_h', 'mean'), MAE_sd=('mae_person_h', 'std'),
        RMSE_mean=('rmse_person_h', 'mean'), RMSE_sd=('rmse_person_h', 'std'),
    ).reset_index().rename(columns={'variant': 'Model'})
    for metric in ('R2', 'MAE', 'RMSE'):
        grouped[f'{metric}_mean±sd'] = grouped.apply(
            lambda r: f'{r[metric+"_mean"]:.4f} ± {r[metric+"_sd"]:.4f}' if r.N > 1 else f'{r[metric+"_mean"]:.4f} (1 run)', axis=1
        )
    grouped.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(grouped[['Model', 'N', 'R2_mean±sd', 'MAE_mean±sd', 'RMSE_mean±sd']].to_string(index=False))
    print('Saved:', args.output)
    print('Warning: same test split + seed count does not prove significance; verify all configs externally.')


if __name__ == '__main__':
    main()

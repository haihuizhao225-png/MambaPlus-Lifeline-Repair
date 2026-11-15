# -*- coding: utf-8 -*-
"""Train / evaluate the revised Frozen-BERT + MambaPlus model.

Supported experiments:
  random_full          10 features: 6 numerical + 3 categorical + BERT text
  random_no_text        9 features: no RepairDescription
  random_no_size        9 features: no RepairSize
  random_no_both        8 features: no RepairSize/RepairDescription
  random_no_element     9 features: no Element; comparator for LOCO
  random_loco_reduced   7 features: no Element/RepairSize/RepairDescription
  loco                 3 leave-one-component-type-out folds, 9 features
  loco_reduced         3 LOCO folds, 7 features
  distance_zzj         near -> far and far -> near, full 10 features
  distance_zyj         near -> far and far -> near, full 10 features
  distance_zzj_reduced, distance_zyj_reduced 8 features

The pretrained BERT encoder is FROZEN: only its pooled vectors are used to
train the tabular MambaPlus model. This source code describes a NEW proposed
experiment, not results already obtained from the legacy scripts.
"""

import argparse
import copy
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import MambaPlusRegressor

NUMERIC = ['RepairSize', 'Sa', 'PGA', 'Mw', 'ZZJ', 'ZYJ']
CATEGORICAL = ['Element', 'ElementMaterial', 'ElementDamage']
TEXT = 'RepairDescription'
TARGET = 'RepairTime'
ALIAS = {
    'RepairSize': ['RepairSize', 'Repair Size', 'Repair_Size'],
    'Sa': ['Sa', 'Bui_Sa_SJ'],
    'PGA': ['PGA', 'Bui_PGA_SJ'],
    'Mw': ['Mw'],
    'ZZJ': ['ZZJ'],
    'ZYJ': ['ZYJ'],
    'Element': ['Element'],
    'ElementMaterial': ['ElementMaterial', 'ElementMaterial_code'],
    'ElementDamage': ['ElementDamage', 'ElementDamage_code'],
    'RepairDescription': ['RepairDescription'],
    'RepairTime': ['RepairTime'],
}
EXPERIMENTS = {
    'random_full': ('random', set()),
    'random_no_text': ('random', {TEXT}),
    'random_no_size': ('random', {'RepairSize'}),
    'random_no_both': ('random', {'RepairSize', TEXT}),
    'random_no_element': ('random', {'Element'}),
    'random_loco_reduced': ('random', {'Element', 'RepairSize', TEXT}),
    'loco': ('loco', set()),
    'loco_reduced': ('loco', {'RepairSize', TEXT}),
    'grouped_full': ('grouped', set()),
    'grouped_reduced': ('grouped', {'RepairSize', TEXT}),
    'distance_zzj': ('zzj', set()),
    'distance_zyj': ('zyj', set()),
    'distance_zzj_reduced': ('zzj', {'RepairSize', TEXT}),
    'distance_zyj_reduced': ('zyj', {'RepairSize', TEXT}),
}


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_data(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in ('.xlsx', '.xlsm', '.xls'):
        df = pd.read_excel(path)
    elif ext in ('.csv', '.txt'):
        try:
            df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='gb18030', low_memory=False)
    else:
        raise ValueError(f'Input must be CSV or Excel, received: {path}')
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for target, aliases in ALIAS.items():
        found = next((name for name in aliases if name in df.columns), None)
        if found is not None and target not in df.columns:
            rename[found] = target
    df = df.rename(columns=rename)
    if TARGET not in df.columns:
        raise KeyError('Missing target column RepairTime')

    original_len = len(df)
    df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
    # No removal of duplicate *feature rows*: repeated engineering records may
    # represent genuinely different repair tasks.
    df = df.loc[df[TARGET].notna() & np.isfinite(df[TARGET]) & (df[TARGET] >= 0)].copy()
    df['_source_row'] = df.index.astype(int)
    df.reset_index(drop=True, inplace=True)
    print(f'[DATA] rows: original={original_len:,}, valid nonnegative target={len(df):,}')
    return df


def feature_columns(df: pd.DataFrame, excluded: set) -> Tuple[List[str], List[str], bool]:
    nums = [f for f in NUMERIC if f not in excluded]
    cats = [f for f in CATEGORICAL if f not in excluded]
    use_text = TEXT not in excluded
    missing = [f for f in nums + cats + ([TEXT] if use_text else []) if f not in df.columns]
    if missing:
        raise KeyError(
            f'Missing columns for this experiment: {missing}. '
            'Frozen-BERT experiments require the ORIGINAL RepairDescription text, '
            'not only the numeric RepairDescription_code column.'
        )
    return nums, cats, use_text


class TrainOnlyPreprocessor:
    """Fitted ONLY on the training subset, separately for each split."""

    def __init__(self, numeric_cols: List[str], categorical_cols: List[str]):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.medians: Dict[str, float] = {}
        self.cat_maps: Dict[str, Dict[str, int]] = {}
        self.feature_scaler: Optional[StandardScaler] = None
        self.target_scaler = StandardScaler()

    @staticmethod
    def _numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _category(series: pd.Series) -> pd.Series:
        return series.fillna('__MISSING__').astype(str).str.strip().replace('', '__MISSING__')

    def fit(self, train: pd.DataFrame):
        if self.numeric_cols:
            cols = []
            for col in self.numeric_cols:
                raw = self._numeric(train[col])
                val = float(raw.median()) if raw.notna().any() else 0.0
                self.medians[col] = val
                cols.append(raw.fillna(val).to_numpy(dtype=np.float64))
            mat = np.stack(cols, axis=1)
            self.feature_scaler = StandardScaler().fit(mat)
        for col in self.categorical_cols:
            labels = sorted(set(self._category(train[col])))
            # 0 reserved for ANY previously unseen or missing label.
            labels = [label for label in labels if label != '__MISSING__']
            self.cat_maps[col] = {label: i + 1 for i, label in enumerate(labels)}
        self.target_scaler.fit(train[[TARGET]].to_numpy(dtype=np.float64))
        return self

    @property
    def cardinalities(self) -> List[int]:
        return [len(self.cat_maps[c]) + 1 for c in self.categorical_cols]

    def transform(self, subset: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.numeric_cols:
            mat = np.stack([
                self._numeric(subset[col]).fillna(self.medians[col]).to_numpy(dtype=np.float64)
                for col in self.numeric_cols
            ], axis=1)
            num = self.feature_scaler.transform(mat).astype(np.float32)
        else:
            num = np.zeros((len(subset), 0), dtype=np.float32)
        if self.categorical_cols:
            cat = np.stack([
                self._category(subset[col]).map(self.cat_maps[col]).fillna(0)
                .to_numpy(dtype=np.int64)
                for col in self.categorical_cols
            ], axis=1)
        else:
            cat = np.zeros((len(subset), 0), dtype=np.int64)
        y = self.target_scaler.transform(subset[[TARGET]].to_numpy(dtype=np.float64))
        return num, cat, y.reshape(-1).astype(np.float32)

    def inverse_target(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).reshape(-1)

    def export(self) -> Dict:
        payload = {
            'numeric_columns': self.numeric_cols,
            'categorical_columns': self.categorical_cols,
            'numeric_medians': self.medians,
            'categorical_maps': self.cat_maps,
            'target_mean': self.target_scaler.mean_.tolist(),
            'target_scale': self.target_scaler.scale_.tolist(),
        }
        if self.feature_scaler is not None:
            payload['numeric_mean'] = self.feature_scaler.mean_.tolist()
            payload['numeric_scale'] = self.feature_scaler.scale_.tolist()
        return payload


class RepairDataset(Dataset):
    def __init__(self, num: np.ndarray, cat: np.ndarray, y: np.ndarray, text: Optional[np.ndarray]):
        self.num = torch.from_numpy(num)
        self.cat = torch.from_numpy(cat)
        self.y = torch.from_numpy(y)
        self.text = torch.from_numpy(text.astype(np.float32)) if text is not None else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        emb = self.text[index] if self.text is not None else torch.empty(0, dtype=torch.float32)
        return self.num[index], self.cat[index], emb, self.y[index]


def compute_frozen_bert_embeddings(df: pd.DataFrame, args) -> np.ndarray:
    """Precompute a fixed pretrained transform; no fitting on target/test data.

    Frozen BERT uses no dataset-derived statistics and receives no target labels.
    Caching its outputs is therefore compatible with disjoint downstream splits.
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('Install transformers: pip install transformers') from exc

    texts = df[TEXT].fillna('').astype(str).tolist()
    digest = hashlib.sha256()
    digest.update(args.bert_model.encode('utf-8'))
    digest.update(str(args.max_tokens).encode('utf-8'))
    for value in texts:
        digest.update(value.encode('utf-8'))
        digest.update(b'\x00')
    short_hash = digest.hexdigest()[:20]
    cache = Path(args.bert_cache) / f'bert_{short_hash}.npy'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not args.refresh_bert_cache:
        array = np.load(cache, allow_pickle=False)
        if array.shape == (len(df), 768):
            print(f'[BERT] loaded frozen vectors: {cache}')
            return array

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, use_fast=True)
    bert = AutoModel.from_pretrained(args.bert_model)
    for p in bert.parameters():
        p.requires_grad_(False)
    device = torch.device(args.device)
    bert = bert.to(device).eval()

    # Actual WordPiece lengths, including special tokens, without truncation.
    lengths = []
    for i in range(0, len(texts), 1000):
        lengths.extend(len(ids) for ids in tokenizer(
            texts[i:i + 1000], padding=False, truncation=False, add_special_tokens=True
        )['input_ids'])
    lengths = np.asarray(lengths)
    summary = {
        'n': len(lengths), 'p50': float(np.percentile(lengths, 50)),
        'p95': float(np.percentile(lengths, 95)),
        'p99': float(np.percentile(lengths, 99)),
        'max': int(lengths.max()),
        'truncated_count': int((lengths > args.max_tokens).sum()),
        'truncated_fraction': float((lengths > args.max_tokens).mean()),
        'maximum_including_special_tokens': args.max_tokens,
    }
    (cache.parent / f'bert_{short_hash}_token_stats.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print('[BERT] token statistics:', summary)
    vectors = []
    with torch.inference_mode():
        for i in range(0, len(texts), args.bert_batch):
            chunk = texts[i:i + args.bert_batch]
            batch = tokenizer(
                chunk, max_length=args.max_tokens, truncation=True,
                padding=True, return_tensors='pt', return_special_tokens_mask=True
            )
            special = batch.pop('special_tokens_mask')
            attention_mask = batch['attention_mask']
            inputs = {key: value.to(device) for key, value in batch.items()}
            encoded = bert(**inputs).last_hidden_state  # [B, token_length, 768]
            # Mean over real content tokens; exclude padding and [CLS]/[SEP].
            content_mask = (attention_mask * (1 - special)).to(device)
            # Empty text: fall back to non-padding special tokens to avoid NaN.
            missing = content_mask.sum(dim=1) == 0
            if missing.any():
                content_mask[missing] = attention_mask.to(device)[missing]
            mask = content_mask.unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            vectors.append(pooled.cpu().numpy().astype(np.float32))
            if (i // args.bert_batch) % 25 == 0:
                print(f'[BERT] processed {min(i + len(chunk), len(texts)):,}/{len(texts):,}')
    array = np.concatenate(vectors, axis=0)
    assert array.shape == (len(df), 768), array.shape
    np.save(cache, array)
    print(f'[BERT] saved frozen vectors: {cache}')
    return array


def make_splits(df: pd.DataFrame, experiment: str, group_col: str, seed: int, independence_col: Optional[str] = None):
    strategy, _ = EXPERIMENTS[experiment]
    n = len(df)
    idx = np.arange(n)
    if strategy == 'random':
        train_val, test = train_test_split(idx, test_size=0.20, random_state=seed, shuffle=True)
        train, val = train_test_split(train_val, test_size=0.125, random_state=seed, shuffle=True)
        return [('random_70_10_20', np.asarray(train), np.asarray(val), np.asarray(test))]

    if strategy == 'grouped':
        if not independence_col or independence_col not in df.columns:
            raise ValueError('For grouped validation provide --independence-col with a '
                             'real facility/site/event identifier column in the dataset.')
        if df[independence_col].isna().any():
            raise ValueError(f'{independence_col} contains missing group identifiers; '
                             'resolve them before grouped validation.')
        grouping = df[independence_col].astype(str).to_numpy()
        if len(np.unique(grouping)) < 3:
            raise ValueError(f'{independence_col} contains fewer than 3 independent groups; '
                             'cannot create group-disjoint train/val/test sets.')
        outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        dev_rel, test_rel = next(outer.split(idx, groups=grouping))
        dev, test = idx[dev_rel], idx[test_rel]
        inner = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
        train_rel, val_rel = next(inner.split(dev, groups=grouping[dev]))
        train, val = dev[train_rel], dev[val_rel]
        return [('group_independent', train, val, test)]

    if strategy == 'loco':
        if group_col not in df.columns:
            raise KeyError(f'Grouping column "{group_col}" not found; pass --component-col')
        groups = df[group_col].fillna('__MISSING__').astype(str).str.strip()
        values = list(groups.value_counts().index)
        if len(values) != 3:
            raise ValueError(
                f'LOCO requires THREE intended facility categories, but {group_col} has '
                f'{len(values)}: {groups.value_counts().head(20).to_dict()}. '
                'Pass --component-col to the column identifying Sewerage/Water Supply/Storm Water, '
                'or filter your dataset to the intended three groups first.'
            )
        folds = []
        for held_out in values:
            test = idx[(groups == held_out).to_numpy()]
            dev = idx[(groups != held_out).to_numpy()]
            train, val = train_test_split(dev, test_size=0.125, random_state=seed, shuffle=True)
            name = 'holdout_' + ''.join(c if c.isalnum() else '_' for c in held_out)[:35]
            folds.append((name, np.asarray(train), np.asarray(val), np.asarray(test)))
        return folds

    distance_feature = 'ZZJ' if strategy == 'zzj' else 'ZYJ'
    distances = pd.to_numeric(df[distance_feature], errors='coerce')
    if distances.isna().any():
        raise ValueError(f'{distance_feature} contains missing/non-numeric values; decide a '
                         'distance grouping policy before running transfer validation.')
    try:
        quartile = pd.qcut(distances, q=4, labels=False, duplicates='drop')
    except ValueError as exc:
        raise ValueError(f'Cannot create distance quartiles for {distance_feature}: {exc}') from exc
    if quartile.nunique() != 4:
        raise ValueError(f'{distance_feature} produced fewer than 4 quartiles due to ties; '
                         'inspect distance distributions rather than splitting equal distances arbitrarily.')
    near = idx[(quartile < 2).to_numpy()]
    far = idx[(quartile >= 2).to_numpy()]
    out = []
    for name, dev, test in [('near_to_far', near, far), ('far_to_near', far, near)]:
        train, val = train_test_split(dev, test_size=0.125, random_state=seed, shuffle=True)
        out.append((f'{distance_feature}_{name}', np.asarray(train), np.asarray(val), np.asarray(test)))
    print(f'[SPLIT] {distance_feature}: near={len(near):,}, far={len(far):,}')
    return out


def build_loader(dataset: RepairDataset, batch: int, shuffle: bool, device: torch.device) -> DataLoader:
    # Always keep the final partial batch, particularly for validation and test.
    return DataLoader(
        dataset, batch_size=batch, shuffle=shuffle, drop_last=False,
        num_workers=0, pin_memory=(device.type == 'cuda'),
    )


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device,
    preprocessor: TrainOnlyPreprocessor,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    preds, actuals = [], []
    with torch.inference_mode():
        for num, cat, txt, y in loader:
            num = num.to(device, non_blocking=True)
            cat = cat.to(device, non_blocking=True)
            text = txt.to(device, non_blocking=True) if txt.shape[-1] == 768 else None
            output = model(num, cat, text)
            preds.append(output.cpu().numpy())
            actuals.append(y.numpy())
    pred_scale = np.concatenate(preds)
    true_scale = np.concatenate(actuals)
    pred = preprocessor.inverse_target(pred_scale)
    true = preprocessor.inverse_target(true_scale)
    metrics = {
        'r2': float(r2_score(true, pred)) if len(true) >= 2 else float('nan'),
        'mae_person_h': float(mean_absolute_error(true, pred)),
        'rmse_person_h': float(np.sqrt(mean_squared_error(true, pred))),
    }
    return metrics, true, pred


def safe_json_write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def _sync_device(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def inference_forward_ms_per_sample(model: nn.Module, loader: DataLoader,
                                    device: torch.device, with_text: bool) -> float:
    """Model-only forward latency on held-out examples, excluding BERT encoding.

    This measure excludes preprocessing, host-to-device copies, and BERT feature
    extraction. Report them separately when evaluating end-to-end deployment.
    """
    model.eval()
    with torch.inference_mode():
        # Warmup, not counted; avoids CUDA first-call initialization overhead.
        first = next(iter(loader))
        num, cat, txt, _ = first
        model(num.to(device), cat.to(device), txt.to(device) if with_text else None)
        _sync_device(device)
        elapsed = 0.0
        count = 0
        for num, cat, txt, _ in loader:
            num = num.to(device, non_blocking=True)
            cat = cat.to(device, non_blocking=True)
            txt = txt.to(device, non_blocking=True) if with_text else None
            _sync_device(device)
            started = time.perf_counter()
            model(num, cat, txt)
            _sync_device(device)
            elapsed += time.perf_counter() - started
            count += len(num)
    return (1000.0 * elapsed / count) if count else float('nan')


def one_run(
    df: pd.DataFrame,
    bert_vectors: Optional[np.ndarray],
    split,
    experiment: str,
    variant: str,
    train_seed: int,
    args,
) -> Dict:
    fold, train_idx, val_idx, test_idx = split
    _, configured_exclusions = EXPERIMENTS[experiment]
    excluded = set(configured_exclusions)
    # Avoid direct disclosure of held-out infrastructure class if the group
    # identifier coincides with one of the selected predictor fields.
    if experiment.startswith('loco') and args.component_col in CATEGORICAL + NUMERIC:
        excluded.add(args.component_col)
    numeric_cols, cat_cols, with_text = feature_columns(df, excluded)
    if with_text and bert_vectors is None:
        raise RuntimeError('Frozen BERT embeddings are required for a text-enabled experiment.')
    if np.intersect1d(train_idx, val_idx).size or np.intersect1d(train_idx, test_idx).size or np.intersect1d(val_idx, test_idx).size:
        raise AssertionError('Training, validation and test indices must be disjoint!')

    fix_seed(train_seed)
    train_df, val_df, test_df = df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]
    preprocessor = TrainOnlyPreprocessor(numeric_cols, cat_cols).fit(train_df)
    train_num, train_cat, train_y = preprocessor.transform(train_df)
    val_num, val_cat, val_y = preprocessor.transform(val_df)
    test_num, test_cat, test_y = preprocessor.transform(test_df)
    selected_bert = lambda ids: bert_vectors[ids] if with_text else None
    train_data = RepairDataset(train_num, train_cat, train_y, selected_bert(train_idx))
    val_data = RepairDataset(val_num, val_cat, val_y, selected_bert(val_idx))
    test_data = RepairDataset(test_num, test_cat, test_y, selected_bert(test_idx))
    device = torch.device(args.device)
    train_loader = build_loader(train_data, args.batch_size, True, device)
    val_loader = build_loader(val_data, args.batch_size, False, device)
    test_loader = build_loader(test_data, args.batch_size, False, device)

    model_kwargs = {
        'n_numerical': len(numeric_cols),
        'categorical_cardinalities': preprocessor.cardinalities,
        'with_text': with_text,
        'variant': variant,
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'dropout': args.dropout,
        'd_state': args.d_state,
        'expand': args.expand,
        'd_conv': args.d_conv,
        'feature_order': args.feature_order,
        'order_seed': args.order_seed,
    }
    model = MambaPlusRegressor(**model_kwargs).to(device)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_model_parameters = sum(p.numel() for p in model.parameters())
    criterion = nn.SmoothL1Loss(beta=args.smooth_l1_beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    order_label = args.feature_order if args.feature_order != 'shuffled' else f'shuffled_{args.order_seed}'
    run_dir = Path(args.output) / experiment / variant / order_label / fold / f'seed_{train_seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    print('\n' + '=' * 90)
    print(f'[RUN] {experiment} | {variant} | {fold} | train seed={train_seed}')
    print(f'      N train/val/test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}; '
          f'num={numeric_cols}; cat={cat_cols}; frozen BERT={with_text}; '
          f'Mamba sequence length={model.tokenizer.n_tokens}')
    print('=' * 90)
    safe_json_write(run_dir / 'preprocessing.json', preprocessor.export())

    best_r2 = -float('inf')
    best_state = None
    best_epoch = None
    no_improve = 0
    epoch_rows = []
    _sync_device(device)
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        for num, cat, txt, y in train_loader:
            num = num.to(device, non_blocking=True)
            cat = cat.to(device, non_blocking=True)
            txt_input = txt.to(device, non_blocking=True) if with_text else None
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(num, cat, txt_input)
            loss = criterion(pred, y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss. Check input values and learning rate.')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_sum += loss.item() * len(y)
            n_seen += len(y)
        scheduler.step()
        val_metrics, _, _ = evaluate(model, val_loader, device, preprocessor)
        val_r2 = val_metrics['r2']
        epoch_rows.append({
            'epoch': epoch, 'train_loss': loss_sum / max(n_seen, 1),
            'validation_r2': val_r2, 'validation_mae_person_h': val_metrics['mae_person_h'],
            'lr': optimizer.param_groups[0]['lr'],
        })
        if np.isfinite(val_r2) and val_r2 > best_r2 + args.min_delta:
            best_r2 = val_r2
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
        if epoch == 1 or epoch % args.print_every == 0 or no_improve >= args.patience:
            print(f'[EPOCH {epoch:3d}] loss={loss_sum / max(n_seen, 1):.5f} '
                  f'val R²={val_r2:.5f} val MAE={val_metrics["mae_person_h"]:.5f} '
                  f'best R²={best_r2:.5f}')
        if no_improve >= args.patience:
            print(f'[EARLY STOP] epoch={epoch}, best_epoch={best_epoch}')
            break
    if best_state is None:
        raise RuntimeError('Could not select a finite-R² checkpoint; inspect target distribution.')
    _sync_device(device)
    training_wall_seconds = time.perf_counter() - training_started
    model.load_state_dict(best_state)
    test_metrics, y_true, y_pred = evaluate(model, test_loader, device, preprocessor)
    print('[TEST] ', test_metrics)
    inference_ms = inference_forward_ms_per_sample(model, test_loader, device, with_text)
    print(f'[COST] trainable parameters={trainable_parameters / 1e6:.3f}M, '
          f'training time={training_wall_seconds:.2f}s, '
          f'model-only inference={inference_ms:.5f}ms/sample (frozen BERT excluded)')

    torch.save({
        'state_dict': best_state,
        'model_kwargs': model_kwargs,
        'feature_columns': {'numerical': numeric_cols, 'categorical': cat_cols, 'text': TEXT if with_text else None},
        'best_epoch': best_epoch,
        'bert_model': args.bert_model if with_text else None,
        'bert_max_tokens': args.max_tokens if with_text else None,
        'text_pooling': 'masked mean excluding special tokens' if with_text else None,
    }, run_dir / 'best_model.pt')
    pd.DataFrame(epoch_rows).to_csv(run_dir / 'training_log.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame({
        'source_row': test_df['_source_row'].to_numpy(),
        'reference_RepairTime_person_h': y_true,
        'predicted_RepairTime_person_h': y_pred,
        'abs_error_person_h': np.abs(y_true - y_pred),
    }).to_csv(run_dir / 'test_predictions.csv', index=False, encoding='utf-8-sig')

    # Independent test-set error by true labor-hour intervals; binning does not train the model.
    bins = pd.cut(y_true, [-np.inf, 8, 24, 100, np.inf], labels=['<=8', '8-24', '24-100', '>100'])
    strata = []
    for level in ['<=8', '8-24', '24-100', '>100']:
        mask = np.asarray(bins == level)
        if mask.sum():
            strata.append({
                'interval_person_h': level, 'n_test': int(mask.sum()),
                'mae_person_h': float(mean_absolute_error(y_true[mask], y_pred[mask])),
                'rmse_person_h': float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
            })
    pd.DataFrame(strata).to_csv(run_dir / 'labor_hour_strata.csv', index=False, encoding='utf-8-sig')
    result = {
        'experiment': experiment, 'variant': variant, 'feature_order': args.feature_order,
        'order_seed': args.order_seed if args.feature_order == 'shuffled' else None,
        'fold': fold,
        'train_seed': train_seed, 'n_train': len(train_idx), 'n_val': len(val_idx),
        'n_test': len(test_idx), 'best_epoch': best_epoch,
        'sequence_length_feature_tokens': model.tokenizer.n_tokens,
        'enabled_modules': ','.join(model.enabled_modules),
        'trainable_parameters_m': trainable_parameters / 1e6,
        'model_parameters_m': all_model_parameters / 1e6,
        'training_wall_seconds_excluding_bert': training_wall_seconds,
        'model_only_inference_ms_per_sample': inference_ms,
        'frozen_bert': with_text, **test_metrics,
    }
    safe_json_write(run_dir / 'metrics.json', result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description='Frozen BERT + real feature-sequence MambaPlus; leakage-safe validation'
    )
    parser.add_argument('--data', type=str, required=True, help='Original CSV/XLSX with RepairTime and RepairDescription')
    parser.add_argument('--output', default='results_mambaplus', help='Directory for result files')
    parser.add_argument('--experiment', default='random_full',
                        choices=sorted(list(EXPERIMENTS) + ['all']),
                        help='One experiment or all available configurations')
    parser.add_argument('--variants', default='mambaplus',
                        help='Comma-separated from mamba,m1,m2,m3,m1_m2,m1_m3,m2_m3,mambaplus or all')
    parser.add_argument('--component-col', default='System',
                        help='Column holding three infrastructure systems for LOCO (often System, not Element)')
    parser.add_argument('--expected-samples', type=int, default=None,
                        help='Optional strict check of final positive-target sample count; e.g. 14695')
    parser.add_argument('--facility-values', default=None,
                        help='Comma-separated names of the THREE selected facilities; '
                             'use when the source includes other facility categories')
    parser.add_argument('--independence-col', default=None,
                        help='Optional grouping key for grouped_full/grouped_reduced (facility ID, site ID, or event ID)')
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--seeds', default='42,52,62,72,82',
                        help='Independent model-training seeds (fixed split seed)')
    parser.add_argument('--bert-model', default='bert-base-multilingual-cased')
    parser.add_argument('--max-tokens', type=int, default=128, help='INCLUDES special tokens')
    parser.add_argument('--bert-batch', type=int, default=32)
    parser.add_argument('--bert-cache', default='bert_embedding_cache')
    parser.add_argument('--refresh-bert-cache', action='store_true')
    parser.add_argument('--device', default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--d-model', type=int, default=384)
    parser.add_argument('--feature-order', choices=['normal', 'reverse', 'shuffled'], default='normal',
                        help='For feature-order sensitivity, retrain with reverse/shuffled order')
    parser.add_argument('--order-seed', type=int, default=2026)
    parser.add_argument('--n-layers', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.025)
    parser.add_argument('--d-state', type=int, default=16)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--d-conv', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--lr', type=float, default=0.00015)
    parser.add_argument('--weight-decay', type=float, default=1e-7)
    parser.add_argument('--smooth-l1-beta', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=80)
    parser.add_argument('--min-delta', type=float, default=1e-5)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--print-every', type=int, default=10)
    args = parser.parse_args()
    args.variants = [x.strip().lower() for x in args.variants.split(',') if x.strip()]
    if args.variants == ['all']:
        args.variants = list(MambaPlusRegressor.ABLATION_VARIANTS)
    if set(args.variants) - set(MambaPlusRegressor.ABLATION_VARIANTS):
        parser.error('Invalid --variants; see --help for the eight allowed combinations')
    try:
        args.seeds = [int(x.strip()) for x in args.seeds.split(',') if x.strip()]
    except ValueError:
        parser.error('--seeds must be comma-separated integers')
    if not args.seeds:
        parser.error('Provide at least one training seed')
    if args.max_tokens < 4 or args.max_tokens > 512:
        parser.error('--max-tokens must be between 4 and 512 for the chosen BERT model')
    return args


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu or fix CUDA PyTorch.')
    df = read_data(args.data)
    # Select the target three facility categories BEFORE any data split.
    if args.facility_values:
        if args.component_col not in df.columns:
            raise KeyError(f'Cannot filter facilities: missing column {args.component_col}')
        selected = [x.strip() for x in args.facility_values.split(',') if x.strip()]
        if len(selected) != 3:
            raise ValueError('--facility-values must specify precisely three facility types')
        group_values = df[args.component_col].fillna('').astype(str).str.strip()
        mask = group_values.str.casefold().isin([x.casefold() for x in selected])
        df = df.loc[mask].reset_index(drop=True)
        observed = set(df[args.component_col].astype(str).str.strip().str.casefold())
        if observed != set(x.casefold() for x in selected):
            raise ValueError(f'Expected these 3 types: {selected}; observed: '
                             f'{df[args.component_col].value_counts().to_dict()}. '
                             'Check facility names and --component-col.')
        print(f'[DATA] retained 3 facility categories: {df[args.component_col].value_counts().to_dict()}')
    elif args.component_col in df.columns and df[args.component_col].nunique() > 3:
        raise ValueError(
            f"The paper evaluates exactly three facility categories, but '{args.component_col}' "
            f"contains {df[args.component_col].nunique()} categories. Specify --facility-values "
            '(e.g. Sewerage,Water Supply,Storm Water) or choose the actual component-group '
            'column with --component-col to avoid silently training on extra facilities.'
        )
    if args.expected_samples is not None and len(df) != args.expected_samples:
        raise ValueError(f'Expected {args.expected_samples} selected nonnegative-target records, '
                         f'but obtained {len(df)}; check missing values, source data and component filtering.')
    requested = list(EXPERIMENTS) if args.experiment == 'all' else [args.experiment]
    for exp in requested:
        _, exclusions = EXPERIMENTS[exp]
        excluded = set(exclusions)
        if exp.startswith('loco') and args.component_col in CATEGORICAL + NUMERIC:
            excluded.add(args.component_col)
        feature_columns(df, excluded)
    needs_bert = any(TEXT not in EXPERIMENTS[exp][1] for exp in requested)
    bert_vectors = compute_frozen_bert_embeddings(df, args) if needs_bert else None
    results = []
    for experiment in requested:
        folds = make_splits(df, experiment, args.component_col, args.split_seed, args.independence_col)
        for split in folds:
            for variant in args.variants:
                for seed in args.seeds:
                    results.append(one_run(df, bert_vectors, split, experiment, variant, seed, args))
                    df_results = pd.DataFrame(results)
                    Path(args.output).mkdir(parents=True, exist_ok=True)
                    df_results.to_csv(Path(args.output) / 'all_run_metrics.csv', index=False, encoding='utf-8-sig')
    raw = pd.DataFrame(results)
    summary = raw.groupby(['experiment', 'variant', 'feature_order', 'order_seed', 'fold'], dropna=False).agg(
        repetitions=('r2', 'size'),
        r2_mean=('r2', 'mean'), r2_sd=('r2', 'std'),
        mae_mean=('mae_person_h', 'mean'), mae_sd=('mae_person_h', 'std'),
        rmse_mean=('rmse_person_h', 'mean'), rmse_sd=('rmse_person_h', 'std'),
    ).reset_index()
    summary.to_csv(Path(args.output) / 'summary_mean_sd.csv', index=False, encoding='utf-8-sig')
    print('\n[COMPLETE] results:', Path(args.output).resolve())
    print(summary.to_string(index=False))
    print('Note: these are newly computed results; do not mix with legacy code metrics.')


if __name__ == '__main__':
    main()

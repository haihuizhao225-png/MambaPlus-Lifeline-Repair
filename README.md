# MambaPlus for Post-Earthquake Lifeline Repair Labor-Hour Estimation

A **research reference implementation** of the heterogeneous tabular MambaPlus framework in the accompanying manuscript. It implements the full 10-field MambaPlus architecture, its M1/M2/M3 modules, eight full-factorial ablations, frozen multilingual BERT encoding, and the manuscript's model-evaluation settings.

**Read [`README_中文.md`](README_中文.md) for Chinese setup and detailed commands.**

## Scope, fidelity and reproducibility

- **Target:** `RepairTime` is *estimated labor effort in person-hours*, derived from repair expenditure; it is **not elapsed calendar repair time**. The script reads this existing field and does not reconstruct it from costs.
- **Inputs:** numerical `RepairSize`, `Sa`, `PGA`, `Mw`, `ZZJ`, `ZYJ`; categorical `Element`, `ElementMaterial`, `ElementDamage`; raw text `RepairDescription`.
- **Text:** frozen `bert-base-multilingual-cased`, up to 128 WordPiece tokens (including special tokens), masked-mean pooling excluding padding and special tokens, 768-dimensional vector, trainable projection to 384 dimensions. **One text field becomes one Mamba feature token**.
- **Mamba:** 8 selective-SSM residual blocks (hidden 384, state 16, expansion 2, depthwise convolution kernel 4), using a **reference pure-PyTorch selective recurrence**, not the optimized `mamba-ssm` fused CUDA kernel. Its throughput and numerical behavior are not claimed to match a particular official Mamba release.
- **Modules:** M1 has three residual MLP branches of widths 384/192/96; M2 applies sample- and feature-token-dependent sigmoid gating; M3 uses four-head field attention, channel attention, and a learnable field-pair relation matrix. All branch outputs remain `[batch, field_tokens, hidden_dim]` before final pooling.
- **Post-fusion:** cross-level weighted skip fusion, sample-conditioned LayerNorm, FFN, and residual refinement, consistent with Fig. 1. The paper's diagram does not uniquely specify all post-fusion equations; the exact implementation choice is documented in `按论文图1至图4的实现说明.md`.
- **Data protocol:** one shared record-level random split (70%/10%/20%, seed 42) and five independent training seeds (42, 52, 62, 72, 82); train-only imputers, scalers, and category mappings; validation-based early stopping; test set withheld from model selection.
- **Limitations:** original repair records, CUDA GPU execution, downloaded BERT weights, and 20-trial random hyperparameter searches are **not included** and have not been reproduced by this package. No manuscript R²/MAE/RMSE values are hard-coded or claimed as outputs. Validation here covers formulas, gradients and a small **synthetic-data** training run.

### Important feature-availability warning

`RepairSize` and `RepairDescription` describe realized/completed repairs and are potentially unavailable at early post-earthquake decision time. Thus the **10-feature experiment is retrospective complete-information estimation**, not verified early-stage forecasting. The eight-feature variant removes both but `ElementDamage` may still require inspection.

## Installation

Recommended: Python 3.10–3.12. On a machine with an NVIDIA RTX 5070 Laptop GPU, install a current CUDA-enabled PyTorch build using the [official PyTorch instructions](https://pytorch.org/get-started/locally/) before installing requirements.

```bash
python -m pip install -r requirements.txt
python test_chapter3_alignment.py
python test_smoke.py
python test_baseline_pipeline.py
```

The third test checks only the optional baseline *pipeline* with a fake estimator; it does **not** verify official Mambular model behavior. Frozen BERT requires online pretrained weights on first use, or a local path through `--bert-model`.

## Example: full-information main experiment

Use a real CSV/XLSX file and specify the actual column containing the **three infrastructure system labels**, often `System`. `Element` denotes a component field in the paper and **should not automatically be treated as the system identifier**.

```bash
python train.py --data "path/to/lifeline.csv" \
  --component-col System \
  --facility-values "Sewerage,Water Supply,Storm Water" \
  --expected-samples 14695 \
  --experiment random_full --variants mambaplus \
  --seeds 42,52,62,72,82
```

`--expected-samples` is a strict check; omit it for a different prepared dataset. **Do not** include genuine sensitive repair records in a public repository.

### Eight factorial M1/M2/M3 combinations

```bash
python train.py --data "path/to/lifeline.csv" --experiment random_full --variants all
```

`all` expands to `mamba`, `m1`, `m2`, `m3`, `m1_m2`, `m1_m3`, `m2_m3`, and `mambaplus`. Module names correspond to **independently enabled modules**, not a cumulative stage number. For an ablation lacking an earlier module, the next enabled module uses the latest available representation (the Mamba output if none is enabled). This is a specified implementation choice, not a demonstrated unique reproduction of historic ablation results.

### Repair information removal

Run separately to isolate the effect of post-repair variables:

```bash
python train.py --data "path/to/lifeline.csv" --experiment random_full
python train.py --data "path/to/lifeline.csv" --experiment random_no_text
python train.py --data "path/to/lifeline.csv" --experiment random_no_size
python train.py --data "path/to/lifeline.csv" --experiment random_no_both
```

The test-prediction CSV is also evaluated in four bins **defined by true person-hours**: `<=8`, `(8,24]`, `(24,100]`, `>100`. Bins are not used for training; each run saves the corresponding sample counts, MAE and RMSE.

### Feature order and distribution shifts

```bash
python train.py --data "path/to/lifeline.csv" --experiment random_full --feature-order reverse
python train.py --data "path/to/lifeline.csv" --experiment random_full --feature-order shuffled --order-seed 1
python train.py --data "path/to/lifeline.csv" --experiment loco --component-col System
python train.py --data "path/to/lifeline.csv" --experiment loco_reduced --component-col System
python train.py --data "path/to/lifeline.csv" --experiment distance_zzj
python train.py --data "path/to/lifeline.csv" --experiment distance_zyj
```

Use five different shuffle seeds plus normal/reverse to cover seven token arrangements. Distance validation includes both near-to-far and far-to-near splits. LOCO uses all 10/8 features if the system label is a separate `System` column; if the system label is itself a model input (e.g. `Element`), it is removed to prevent direct group-label leakage (9/7 features). For facility/site/event-independent splitting, provide a real identifier and run `--experiment grouped_full --independence-col <column>`.

## Results and computational accounting

Results are saved under `results_mambaplus/` by experiment, ablation, ordering, fold and seed. Outputs include `best_model.pt`, `preprocessing.json`, `training_log.csv`, `test_predictions.csv`, `labor_hour_strata.csv`, `metrics.json`, `all_run_metrics.csv`, and `summary_mean_sd.csv`.

The code reports **trainable parameters**, training wall time and **model-forward-only** inference milliseconds/sample. The timing **excludes frozen BERT embedding precomputation**, preprocessing and host/device data transfer; report these separately for any end-to-end deployment comparison. The optional `baseline_mamba_models.py` uses the upstream `mambular` library for MambaTab/Mambular regression; install `requirements_baselines.txt` and check upstream API compatibility. The package does not implement every other baseline described in the article.

## Publishing

See [`GITHUB_UPLOAD.md`](GITHUB_UPLOAD.md) for instructions for creating a private GitHub repository from this folder. Upload is not complete until you authenticate and push it; the package deliberately contains no credentials or real datasets. No code license has been assumed on the author's behalf.

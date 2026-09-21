# Optimal Pore Pressure Prediction via Bayesian Decision-Tree Meta-Learning

Source code for the paper *"Optimal Pore Pressure Prediction via Bayesian
Decision-Tree Meta-Learning"*. It benchmarks eleven machine-learning models for
pore-pressure prediction across three geological zones of Pakistan's Murree
Formation, using a blind-well design (one well each for training, validation and
blind testing), Bayesian (Optuna/TPE) hyperparameter optimization, and
decision-tree / linear-regression meta-learner stacking.

## Contents

| File | Purpose |
|------|---------|
| `all_baselines.py` | Trains and tunes all base models (Decision Tree, AdaBoost, Random Forest, XGBoost, LightGBM, CatBoost, FFNN, CNN, LSTM/GRU, Transformer) and scores them on the blind well. |
| `optimal_five_good_baselines.py` | Focused run on the best-performing baselines, with the paper's figures. |
| `ablantion_studies.py` | Exhaustive stacking ablation over combinations of positive-R² base models. |
| `example_quicktest.py` | Small, self-contained example of the pipeline on synthetic data (see below). |
| `LICENSE` | MIT license. |

The full scripts were developed on Google Colab and expect the well-log CSV files
(`MISSA-KESWAL-01/02/03.CSV`, etc.) in the working directory. The well data are
proprietary and are not distributed here.

## Requirements

- Python 3.10+
- `numpy`, `scikit-learn`, `optuna` (for the quick test)
- Additionally `xgboost`, `lightgbm`, `catboost`, `torch`, `pytorch_lightning`
  (for the full baselines)

```bash
pip install numpy scikit-learn optuna
```

## Quick test

`example_quicktest.py` reproduces the core method — Optuna hyperparameter search,
multi-seed retraining, blind-well scoring, and meta-decision-tree stacking — on
small synthetic well-log data. It needs no proprietary data and no GPU, and runs
in a few seconds.

```bash
python example_quicktest.py
```

Expected output (values are deterministic given the fixed seeds):

```
Base-model blind-well R2
  DecisionTree : 0.9089
  RandomForest : 0.9666

Meta-decision-tree stack (DT + RF) blind-well R2: 0.9652
Best single base model blind-well R2               : 0.9666

Quick test passed.
```

The script prints `Quick test passed.` and exits 0 when the pipeline runs
correctly.

## License

Released under the MIT License (see `LICENSE`).

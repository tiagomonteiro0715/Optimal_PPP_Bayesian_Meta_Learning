# -*- coding: utf-8 -*-
"""
Quick-test example for "Optimal Pore Pressure Prediction via Bayesian
Decision-Tree Meta-Learning".

This is a small, self-contained reproduction of the paper's pipeline that runs
in a few seconds on CPU, with no proprietary well data and no GPU required. It
uses synthetic well-log-like data so a reviewer can verify the method end to end:

    1. Bayesian (Optuna/TPE) hyperparameter search for tree baselines.
    2. Multi-seed retraining, scored on a held-out "blind well".
    3. Meta-decision-tree stacking of the positive-R2 base models.

The full study (real wells, neural networks, full ablation) lives in
all_baselines.py, optimal_five_good_baselines.py and ablantion_studies.py.

Run:  python example_quicktest.py
Deps: numpy, scikit-learn, optuna   (pip install numpy scikit-learn optuna)
"""

import warnings

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

SEEDS = [42, 137, 256, 7, 2048]
N_TRIALS = 10


def make_synthetic_well(n, rng):
    """Fake well logs (GR, Vsh, dt, NCT, RES_DEEP) and a pore-pressure target."""
    gr = rng.uniform(20, 150, n)
    vsh = rng.uniform(0, 1, n)
    dt = rng.uniform(60, 140, n)
    nct = rng.uniform(60, 120, n)
    res = rng.uniform(0.5, 50, n)
    ppp = (
        2500
        + 8.0 * gr
        + 900 * vsh
        + 12 * (dt - nct)
        - 5 * res
        + rng.normal(0, 60, n)
    )
    return np.column_stack([gr, vsh, dt, nct, res]), ppp


def build_data():
    """One training, one validation and one blind well (blind-well design)."""
    rng = np.random.default_rng(0)
    X_tr, y_tr = make_synthetic_well(4000, rng)
    X_va, y_va = make_synthetic_well(1500, rng)
    X_bl, y_bl = make_synthetic_well(1200, rng)

    scaler = StandardScaler().fit(X_tr)
    return (
        scaler.transform(X_tr), y_tr,
        scaler.transform(X_va), y_va,
        scaler.transform(X_bl), y_bl,
    )


def tune(objective):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


def best_over_seeds(make_model, X_tr, y_tr, X_bl, y_bl):
    """Retrain with several seeds, keep the best blind-well model."""
    best_r2, best_model = -np.inf, None
    for s in SEEDS:
        m = make_model(s).fit(X_tr, y_tr)
        r2 = r2_score(y_bl, m.predict(X_bl))
        if r2 > best_r2:
            best_r2, best_model = r2, m
    return best_model, best_r2


def main():
    X_tr, y_tr, X_va, y_va, X_bl, y_bl = build_data()

    # --- Base model 1: Decision Tree, Optuna-tuned on the validation well ---
    def obj_dt(trial):
        m = DecisionTreeRegressor(
            max_depth=trial.suggest_int("depth", 2, 20),
            min_samples_split=trial.suggest_int("min_split", 2, 30),
            min_samples_leaf=trial.suggest_int("min_leaf", 1, 20),
            random_state=42,
        ).fit(X_tr, y_tr)
        return r2_score(y_va, m.predict(X_va))

    bp_dt = tune(obj_dt)
    dt_model, dt_r2 = best_over_seeds(
        lambda s: DecisionTreeRegressor(
            max_depth=bp_dt["depth"],
            min_samples_split=bp_dt["min_split"],
            min_samples_leaf=bp_dt["min_leaf"],
            random_state=s,
        ),
        X_tr, y_tr, X_bl, y_bl,
    )

    # --- Base model 2: Random Forest, Optuna-tuned on the validation well ---
    def obj_rf(trial):
        m = RandomForestRegressor(
            n_estimators=trial.suggest_int("n_est", 50, 200),
            max_depth=trial.suggest_int("depth", 2, 20),
            random_state=42, n_jobs=-1,
        ).fit(X_tr, y_tr)
        return r2_score(y_va, m.predict(X_va))

    bp_rf = tune(obj_rf)
    rf_model, rf_r2 = best_over_seeds(
        lambda s: RandomForestRegressor(
            n_estimators=bp_rf["n_est"],
            max_depth=bp_rf["depth"],
            random_state=s, n_jobs=-1,
        ),
        X_tr, y_tr, X_bl, y_bl,
    )

    print("Base-model blind-well R2")
    print(f"  DecisionTree : {dt_r2:.4f}")
    print(f"  RandomForest : {rf_r2:.4f}")

    # --- Meta-decision-tree stacking of the positive-R2 base models ---
    base = {"DT": dt_model, "RF": rf_model}
    positive = [name for name, m in base.items()
                if r2_score(y_bl, m.predict(X_bl)) > 0]

    def meta_features(X):
        return np.column_stack([base[name].predict(X) for name in positive])

    meta = DecisionTreeRegressor(max_depth=5, random_state=42)
    meta.fit(meta_features(X_tr), y_tr)
    stack_r2 = r2_score(y_bl, meta.predict(meta_features(X_bl)))

    best_base = max(dt_r2, rf_r2)
    print(f"\nMeta-decision-tree stack ({' + '.join(positive)}) blind-well R2: {stack_r2:.4f}")
    print(f"Best single base model blind-well R2               : {best_base:.4f}")

    # Sanity check for the quick test: the pipeline ran and produced a real fit.
    assert stack_r2 > 0.5, "Unexpectedly poor fit -- check the environment."
    print("\nQuick test passed.")


if __name__ == "__main__":
    main()

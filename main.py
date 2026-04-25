# -*- coding: utf-8 -*-
"""
Active Learning Framework for High-Entropy Perovskite Screening

Core method:
- Gaussian Process Regression (GPR)
- Bayesian Optimization for hyperparameter tuning
- Uncertainty-guided active learning

Note:
This script provides the general workflow used in the manuscript.
Input data is not included.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
from sklearn.preprocessing import MinMaxScaler
from bayes_opt import BayesianOptimization

# Configuration
CONFIG = {
    "initial_data": "Train_data.xlsx",
    "unlabeled_data": "Sample.xlsx",
    "target_label": "EOv",   # switchable: EOv / EN / EH
    "max_iter": 50,
    "random_state": 42,
    "std_threshold": 0.4
}

# Data Loading
def load_data(config):
    df_labeled = pd.read_excel(config["initial_data"])
    df_unlabeled = pd.read_excel(config["unlabeled_data"])

    non_feature_cols = [config["target_label"]]
    feature_cols = [c for c in df_labeled.columns if c not in non_feature_cols]

    X_labeled = df_labeled[feature_cols].values
    y_labeled = df_labeled[config["target_label"]].values
    X_unlabeled = df_unlabeled[feature_cols].values

    return X_labeled, y_labeled, X_unlabeled, feature_cols


# Model + BO optimization
def optimize_gpr(X, y, random_state):
    def objective(alpha_log, length_scale_log):
        alpha = 10 ** alpha_log
        length_scale = 10 ** length_scale_log
        kernel = C(1.0) * Matern(length_scale=length_scale, nu=2.5)

        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=True,
            random_state=random_state
        )

        kf = KFold(n_splits=3, shuffle=True, random_state=random_state)
        y_pred = cross_val_predict(model, X, y, cv=kf)
        return -mean_squared_error(y, y_pred)

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={
            "alpha_log": (-8, -4),
            "length_scale_log": (-2, 2)
        },
        random_state=random_state,
        verbose=0
    )

    optimizer.maximize(init_points=5, n_iter=10)
    params = optimizer.max["params"]

    alpha = 10 ** params["alpha_log"]
    length_scale = 10 ** params["length_scale_log"]

    kernel = C(1.0) * Matern(length_scale=length_scale, nu=2.5)
    return kernel, alpha

# Active Learning Loop
def active_learning(config):
    X_labeled, y_labeled, X_unlabeled, feature_cols = load_data(config)

    scaler = MinMaxScaler()
    X_all = np.vstack([X_labeled, X_unlabeled])
    X_all = scaler.fit_transform(X_all)

    X_labeled = X_all[:len(X_labeled)]
    X_unlabeled = X_all[len(X_labeled):]

    for i in range(config["max_iter"]):

        kernel, alpha = optimize_gpr(
            X_labeled, y_labeled, config["random_state"] + i
        )

        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=True,
            random_state=config["random_state"] + i
        )

        model.fit(X_labeled, y_labeled)

        # Predict on unlabeled pool
        pred, std = model.predict(X_unlabeled, return_std=True)

        # Acquisition: UCB
        score = pred + 2.0 * std
        idx = np.argmax(score)

        if std[idx] > config["std_threshold"]:
            continue

        # Add pseudo-label
        X_labeled = np.vstack([X_labeled, X_unlabeled[idx]])
        y_labeled = np.append(y_labeled, pred[idx])

        X_unlabeled = np.delete(X_unlabeled, idx, axis=0)

        print(f"Iter {i+1}: added sample, pred={pred[idx]:.3f}, std={std[idx]:.3f}")

    return model

# Main
if __name__ == "__main__":
    model = active_learning(CONFIG)
    print("Active learning finished.")

"""Numerical analysis functions retained from the author-provided n=76 workflow.
No training data, fitted models, acquisition automation, or local paths are embedded.
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Any, Iterable
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler
FEATURES = ("rB", "IE", "ISA", "χ", "Z", "M", "S_config")
TARGETS = ("N_Abs", "H_Abs", "E_Ov")
RANDOM_SEED = 2026
OUTER_SPLITS = 5
INNER_SPLITS = 4
RANDOM_REPEATS = 5
N_RESTARTS_OPTIMIZER = 0
EPS = 1e-12


@dataclass
class FitResult:
    scaler: StandardScaler
    model: GaussianProcessRegressor
    warning_messages: list[str]
    diagnostics: dict[str, Any]

def safe_corr(function, x: Iterable[float], y: Iterable[float]) -> float:
    x_array = np.asarray(list(x), dtype=float)
    y_array = np.asarray(list(y), dtype=float)
    if (
        len(x_array) < 2
        or np.std(x_array) <= EPS
        or np.std(y_array) <= EPS
    ):
        return float("nan")
    return float(function(x_array, y_array).statistic)

def numeric_frame(frame: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    columns = tuple(columns)
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="raise")
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite values in columns: {columns}")
    return array

def make_kernel():
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=1.0,
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        )
        + WhiteKernel(
            noise_level=0.05,
            noise_level_bounds=(1e-8, 1.0),
        )
    )

def boundary_flag(
    value: float,
    bounds: tuple[float, float],
    tolerance: float = 0.02,
) -> str:
    low, high = bounds
    if value <= low * (1.0 + tolerance):
        return "LOWER_BOUND_CONTACT"
    if value >= high * (1.0 - tolerance):
        return "UPPER_BOUND_CONTACT"
    return ""

def fit_gpr(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    context: dict[str, Any],
) -> FitResult:
    scaler = StandardScaler().fit(X)
    model = GaussianProcessRegressor(
        kernel=make_kernel(),
        alpha=1e-10,
        normalize_y=True,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=N_RESTARTS_OPTIMIZER,
        random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(scaler.transform(X), y)
    warning_messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    fitted = model.kernel_
    length_scale = float(
        np.mean(np.asarray(fitted.k1.k2.length_scale, dtype=float))
    )
    constant = float(fitted.k1.k1.constant_value)
    noise = float(fitted.k2.noise_level)
    diagnostics = {
        **context,
        "n_fit_rows": int(len(y)),
        "optimized_kernel": str(fitted),
        "log_marginal_likelihood": float(
            model.log_marginal_likelihood_value_
        ),
        "constant_value": constant,
        "length_scale": length_scale,
        "noise_level": noise,
        "constant_boundary": boundary_flag(
            constant, (1e-3, 1e3)
        ),
        "length_scale_boundary": boundary_flag(
            length_scale, (1e-2, 1e2)
        ),
        "noise_boundary": boundary_flag(
            noise, (1e-8, 1.0)
        ),
        "warning_count": len(warning_messages),
    }
    return FitResult(
        scaler=scaler,
        model=model,
        warning_messages=warning_messages,
        diagnostics=diagnostics,
    )

def predict_gpr(
    fit: FitResult,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean, std = fit.model.predict(
        fit.scaler.transform(X), return_std=True
    )
    return np.asarray(mean, dtype=float), np.maximum(
        np.asarray(std, dtype=float), EPS
    )

def calibrate_sigma(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    uids: np.ndarray,
    scheme: str,
    seed: int,
    context: dict[str, Any],
) -> tuple[
    float,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    prediction = np.full(len(y), np.nan)
    raw_std = np.full(len(y), np.nan)
    audit_rows = []
    diagnostic_rows = []
    warning_rows = []

    if scheme == "group":
        n_splits = min(INNER_SPLITS, len(np.unique(groups)))
        if n_splits < 2:
            raise ValueError(
                "At least two groups are required for calibration."
            )
        splits = GroupKFold(n_splits=n_splits).split(
            X, groups=groups
        )
    else:
        n_splits = min(INNER_SPLITS, len(X))
        splits = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        ).split(X)

    for inner_fold, (fit_idx, validation_idx) in enumerate(
        splits, start=1
    ):
        fit_context = {
            **context,
            "fit_role": "inner_uncertainty_calibration",
            "inner_fold": inner_fold,
        }
        fit = fit_gpr(
            X[fit_idx],
            y[fit_idx],
            seed + inner_fold,
            fit_context,
        )
        prediction[validation_idx], raw_std[validation_idx] = (
            predict_gpr(fit, X[validation_idx])
        )
        diagnostic_rows.append(fit.diagnostics)
        warning_rows.extend(
            [
                {**fit_context, "message": message}
                for message in fit.warning_messages
            ]
        )
        audit_rows.append(
            {
                **context,
                "inner_fold": inner_fold,
                "inner_train_count": len(fit_idx),
                "inner_validation_count": len(validation_idx),
                "inner_train_uid": "|".join(
                    sorted(uids[fit_idx].astype(str))
                ),
                "inner_validation_uid": "|".join(
                    sorted(uids[validation_idx].astype(str))
                ),
                "uid_overlap_count": len(
                    set(uids[fit_idx].astype(str))
                    & set(uids[validation_idx].astype(str))
                ),
                "inner_train_groups": "|".join(
                    sorted(set(groups[fit_idx].astype(str)))
                ),
                "inner_validation_groups": "|".join(
                    sorted(
                        set(groups[validation_idx].astype(str))
                    )
                ),
                "group_overlap_count": len(
                    set(groups[fit_idx].astype(str))
                    & set(groups[validation_idx].astype(str))
                ),
            }
        )

    standardized = (
        y - prediction
    ) / np.maximum(raw_std, EPS)
    scale = float(
        np.clip(
            np.sqrt(np.mean(standardized ** 2)),
            0.25,
            10.0,
        )
    )
    return (
        scale,
        pd.DataFrame(audit_rows),
        pd.DataFrame(diagnostic_rows),
        pd.DataFrame(warning_rows),
    )

def metric_dict(
    truth: Iterable[float],
    prediction: Iterable[float],
    sigma: Iterable[float],
    dummy_prediction: Iterable[float] | None = None,
) -> dict[str, float]:
    truth = np.asarray(list(truth), dtype=float)
    prediction = np.asarray(list(prediction), dtype=float)
    sigma = np.maximum(
        np.asarray(list(sigma), dtype=float), EPS
    )
    residual = prediction - truth
    absolute_error = np.abs(residual)
    rmse = float(
        np.sqrt(mean_squared_error(truth, prediction))
    )
    result = {
        "n_predictions": int(len(truth)),
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(truth, prediction)),
        "R2": float(r2_score(truth, prediction)),
        "Pearson": safe_corr(pearsonr, truth, prediction),
        "Spearman": safe_corr(spearmanr, truth, prediction),
        "bias_prediction_minus_DFT": float(
            np.mean(residual)
        ),
        "max_AE": float(np.max(absolute_error)),
        "Gaussian_NLL": float(
            np.mean(
                0.5 * np.log(2.0 * np.pi * sigma ** 2)
                + 0.5
                * ((truth - prediction) / sigma) ** 2
            )
        ),
        "coverage_68": float(
            np.mean(absolute_error <= sigma)
        ),
        "coverage_95": float(
            np.mean(absolute_error <= 1.96 * sigma)
        ),
        "median_calibrated_std": float(np.median(sigma)),
        "mean_absolute_standardized_residual": float(
            np.mean(absolute_error / sigma)
        ),
        "max_absolute_standardized_residual": float(
            np.max(absolute_error / sigma)
        ),
        "uncertainty_error_Spearman": safe_corr(
            spearmanr, sigma, absolute_error
        ),
    }
    if dummy_prediction is not None:
        dummy_prediction = np.asarray(
            list(dummy_prediction), dtype=float
        )
        dummy_rmse = float(
            np.sqrt(
                mean_squared_error(truth, dummy_prediction)
            )
        )
        result["dummy_RMSE"] = dummy_rmse
        result["RMSE_over_dummy"] = (
            rmse / max(dummy_rmse, EPS)
        )
    return result

def outer_splits(
    X: np.ndarray,
    groups: np.ndarray,
    scheme: str,
    seed: int,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    if scheme == "group":
        n_splits = min(
            OUTER_SPLITS, len(np.unique(groups))
        )
        splitter = GroupKFold(n_splits=n_splits)
        return [
            (
                0,
                fold,
                np.asarray(fit_idx),
                np.asarray(test_idx),
            )
            for fold, (fit_idx, test_idx) in enumerate(
                splitter.split(X, groups=groups)
            )
        ]

    n_splits = min(OUTER_SPLITS, len(X))
    splitter = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=RANDOM_REPEATS,
        random_state=seed,
    )
    output = []
    for split_number, (fit_idx, test_idx) in enumerate(
        splitter.split(X)
    ):
        output.append(
            (
                split_number // n_splits,
                split_number % n_splits,
                np.asarray(fit_idx),
                np.asarray(test_idx),
            )
        )
    return output

def run_cv(
    frame: pd.DataFrame,
    scheme: str,
    cumulative_n: int,
) -> dict[str, pd.DataFrame]:
    X = numeric_frame(frame, FEATURES)
    groups = frame["B_site_elements"].astype(str).to_numpy()
    uids = frame["sample_uid"].astype(str).to_numpy()
    splits = outer_splits(
        X,
        groups,
        scheme,
        RANDOM_SEED + cumulative_n,
    )

    prediction_rows = []
    fold_metric_rows = []
    outer_audit_rows = []
    inner_audit_parts = []
    diagnostic_rows = []
    warning_rows = []

    for target_index, target in enumerate(TARGETS):
        y = pd.to_numeric(
            frame[target], errors="raise"
        ).to_numpy(dtype=float)
        for repeat, outer_fold, fit_idx, test_idx in splits:
            context = {
                "scheme": scheme,
                "cumulative_n": cumulative_n,
                "target": target,
                "repeat": repeat,
                "outer_fold": outer_fold,
            }
            seed = (
                RANDOM_SEED
                + cumulative_n * 100000
                + target_index * 10000
                + repeat * 100
                + outer_fold * 10
            )
            (
                scale,
                inner_audit,
                inner_diagnostics,
                inner_warnings,
            ) = calibrate_sigma(
                X[fit_idx],
                y[fit_idx],
                groups[fit_idx],
                uids[fit_idx],
                scheme,
                seed,
                context,
            )
            inner_audit_parts.append(inner_audit)
            diagnostic_rows.extend(
                inner_diagnostics.to_dict("records")
            )
            warning_rows.extend(
                inner_warnings.to_dict("records")
            )

            outer_context = {
                **context,
                "fit_role": "outer_prediction",
                "inner_fold": 0,
            }
            fit = fit_gpr(
                X[fit_idx],
                y[fit_idx],
                seed + 5000,
                outer_context,
            )
            prediction, raw_std = predict_gpr(
                fit, X[test_idx]
            )
            calibrated_std = raw_std * scale
            dummy = np.repeat(
                np.mean(y[fit_idx]), len(test_idx)
            )
            diagnostics = dict(fit.diagnostics)
            diagnostics["calibration_scale"] = scale
            diagnostic_rows.append(diagnostics)
            warning_rows.extend(
                [
                    {**outer_context, "message": message}
                    for message in fit.warning_messages
                ]
            )

            outer_audit_rows.append(
                {
                    **context,
                    "outer_train_count": len(fit_idx),
                    "outer_test_count": len(test_idx),
                    "outer_train_uid": "|".join(
                        sorted(uids[fit_idx])
                    ),
                    "outer_test_uid": "|".join(
                        sorted(uids[test_idx])
                    ),
                    "uid_overlap_count": len(
                        set(uids[fit_idx])
                        & set(uids[test_idx])
                    ),
                    "outer_train_groups": "|".join(
                        sorted(set(groups[fit_idx]))
                    ),
                    "outer_test_groups": "|".join(
                        sorted(set(groups[test_idx]))
                    ),
                    "group_overlap_count": len(
                        set(groups[fit_idx])
                        & set(groups[test_idx])
                    ),
                    "calibration_scale": scale,
                }
            )
            fold_metric_rows.append(
                {
                    **context,
                    **metric_dict(
                        y[test_idx],
                        prediction,
                        calibrated_std,
                        dummy,
                    ),
                }
            )
            for local_index, row_index in enumerate(
                test_idx
            ):
                prediction_rows.append(
                    {
                        **context,
                        "sample_uid": uids[row_index],
                        "round_added": int(
                            frame.iloc[row_index][
                                "round_added"
                            ]
                        ),
                        "B_site_elements": groups[row_index],
                        "true_DFT": y[row_index],
                        "prediction": prediction[local_index],
                        "raw_std": raw_std[local_index],
                        "calibrated_std": calibrated_std[
                            local_index
                        ],
                        "dummy_prediction": dummy[local_index],
                        "residual_prediction_minus_DFT": (
                            prediction[local_index]
                            - y[row_index]
                        ),
                        "absolute_error": abs(
                            prediction[local_index]
                            - y[row_index]
                        ),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    metric_rows = []
    for target, table in predictions.groupby("target"):
        metric_rows.append(
            {
                "scheme": scheme,
                "cumulative_n": cumulative_n,
                "target": target,
                **metric_dict(
                    table["true_DFT"],
                    table["prediction"],
                    table["calibrated_std"],
                    table["dummy_prediction"],
                ),
            }
        )

    return {
        "metrics": pd.DataFrame(metric_rows),
        "predictions": predictions,
        "fold_metrics": pd.DataFrame(
            fold_metric_rows
        ),
        "outer_audit": pd.DataFrame(
            outer_audit_rows
        ),
        "inner_audit": (
            pd.concat(
                inner_audit_parts, ignore_index=True
            )
            if inner_audit_parts
            else pd.DataFrame()
        ),
        "diagnostics": pd.DataFrame(
            diagnostic_rows
        ),
        "warnings": pd.DataFrame(
            warning_rows
        ),
    }

def full_model_predictions(
    train: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    label: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    X = numeric_frame(train, FEATURES)
    X_prediction = numeric_frame(
        prediction_frame, FEATURES
    )
    groups = train[
        "B_site_elements"
    ].astype(str).to_numpy()
    uids = train[
        "sample_uid"
    ].astype(str).to_numpy()

    output = prediction_frame.copy()
    summary_rows = []
    diagnostic_rows = []
    for target_index, target in enumerate(TARGETS):
        y = pd.to_numeric(
            train[target], errors="raise"
        ).to_numpy(dtype=float)
        context = {
            "scheme": "full_training_deployment",
            "training_label": label,
            "training_n": len(train),
            "target": target,
        }
        seed = (
            RANDOM_SEED
            + len(train) * 1000
            + target_index * 100
        )
        (
            scale,
            _,
            inner_diagnostics,
            _,
        ) = calibrate_sigma(
            X,
            y,
            groups,
            uids,
            "group",
            seed,
            context,
        )
        fit = fit_gpr(
            X,
            y,
            seed + 5000,
            {
                **context,
                "fit_role": "final_deployment",
            },
        )
        mean, raw_std = predict_gpr(
            fit, X_prediction
        )
        sigma = raw_std * scale
        output[f"pred_{target}_{label}"] = mean
        output[
            f"calibrated_std_{target}_{label}"
        ] = sigma
        summary_rows.append(
            {
                "training_label": label,
                "training_n": len(train),
                "target": target,
                "calibration_scale": scale,
                "prediction_set_median_std": float(
                    np.median(sigma)
                ),
                "prediction_set_mean_std": float(
                    np.mean(sigma)
                ),
                "prediction_set_p90_std": float(
                    np.quantile(sigma, 0.90)
                ),
                "prediction_set_p95_std": float(
                    np.quantile(sigma, 0.95)
                ),
                "prediction_set_max_std": float(
                    np.max(sigma)
                ),
            }
        )
        diagnostic_rows.extend(
            inner_diagnostics.to_dict("records")
        )
        diagnostic = dict(fit.diagnostics)
        diagnostic["calibration_scale"] = scale
        diagnostic_rows.append(diagnostic)
    return (
        output,
        pd.DataFrame(summary_rows),
        pd.DataFrame(diagnostic_rows),
    )

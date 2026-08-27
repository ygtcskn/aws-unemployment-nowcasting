"""Estimate a pooled LightGBM model with rolling out-of-sample forecasts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from tqdm import tqdm

from model_utils import (
    add_target_lags,
    evaluate_predictions,
    exclude_target_lag_features,
    load_model_panel,
    load_tuned_config,
    prediction_frame,
    print_evaluation,
    rolling_month_splits,
    save_evaluation,
    select_window,
)


DATA_PATH = Path("data/final/gt_unemp_monthly_final.csv")
OUTPUT_DIR = Path("reports/outputs")
WINDOW_MONTHS = 60
TARGET_LAGS = 1
N_ESTIMATORS = 300
RANDOM_STATE = 42
DEFAULT_MODEL_PARAMS = {
    "n_estimators": N_ESTIMATORS,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 30,
    "min_split_gain": 0.0,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "max_bin": 255,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--window-months", type=int)
    parser.add_argument("--target-lags", type=int)
    parser.add_argument("--n-estimators", type=int)
    parser.add_argument("--params-file", type=Path)
    parser.add_argument("--test-start-date")
    return parser.parse_args()


def make_model(model_params=None):
    params = DEFAULT_MODEL_PARAMS.copy()
    if model_params:
        unknown = set(model_params) - set(params)
        if unknown:
            raise ValueError(f"Unsupported LightGBM parameters: {sorted(unknown)}")
        params.update(model_params)

    return LGBMRegressor(
        objective="regression",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        **params,
    )


def main(
    data_path,
    output_dir,
    window_months=None,
    target_lags=None,
    n_estimators=None,
    params_file=None,
    test_start_date=None,
):
    data, target, features = load_model_panel(data_path)
    features = exclude_target_lag_features(features, target)
    tuned_params = {}
    tuned_test_start = None
    if params_file is not None:
        tuned_params, tuned_test_start, _ = load_tuned_config(
            params_file, "lightgbm", target
        )

    tuned_window = tuned_params.pop("window_months", None)
    tuned_lags = tuned_params.pop("target_lags", None)
    window_months = (
        window_months
        if window_months is not None
        else tuned_window if tuned_window is not None else WINDOW_MONTHS
    )
    target_lags = (
        target_lags
        if target_lags is not None
        else tuned_lags if tuned_lags is not None else TARGET_LAGS
    )
    model_params = DEFAULT_MODEL_PARAMS.copy()
    model_params.update(tuned_params)
    if n_estimators is not None:
        model_params["n_estimators"] = n_estimators
    if model_params["n_estimators"] < 1:
        raise ValueError("n_estimators must be positive.")

    if test_start_date is not None:
        test_start = pd.Timestamp(test_start_date)
        if pd.isna(test_start):
            raise ValueError("test_start_date is invalid.")
    else:
        test_start = tuned_test_start
    if (
        tuned_test_start is not None
        and test_start is not None
        and test_start < tuned_test_start
    ):
        raise ValueError(
            "test_start_date cannot precede the holdout start saved by Optuna."
        )

    data, lag_columns = add_target_lags(data, target, target_lags)
    features = list(dict.fromkeys(features + lag_columns))
    splits = rolling_month_splits(data, window_months)
    if test_start is not None:
        splits = [split for split in splits if pd.Timestamp(split[1]) >= test_start]
        if not splits:
            raise ValueError("No forecast months remain on or after test_start_date.")

    predictions = []
    importances = []
    final_model = None

    for train_months, test_month in tqdm(
        splits, desc="Rolling LightGBM", unit="month"
    ):
        train, test = select_window(data, train_months, test_month)
        train = train.dropna(subset=features + [target]).copy()
        test = test.dropna(subset=features + [target]).copy()
        if train.empty or test.empty:
            continue

        model = make_model(model_params)
        model.fit(train[features], train[target])
        forecast = model.predict(test[features])
        predictions.append(
            prediction_frame(
                test, test[target], forecast, "LightGBM", train_months
            )
        )

        gain = model.booster_.feature_importance(importance_type="gain")
        if gain.sum() > 0:
            gain = gain / gain.sum()
        importances.append(gain)
        final_model = model

    if not predictions or final_model is None:
        raise ValueError("No rolling LightGBM predictions were produced.")

    predictions = pd.concat(predictions, ignore_index=True)
    predictions, overall, by_country = evaluate_predictions(predictions)
    paths = save_evaluation(
        predictions, overall, by_country, output_dir, prefix="lightgbm"
    )

    importance_matrix = np.vstack(importances)
    importance = pd.DataFrame(
        {
            "feature": features,
            "mean_importance": importance_matrix.mean(axis=0),
            "importance_std": importance_matrix.std(axis=0),
        }
    ).sort_values("mean_importance", ascending=False)
    importance_path = Path(output_dir) / "lightgbm_feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    model_path = Path(output_dir) / "lightgbm_last_window.txt"
    final_model.booster_.save_model(model_path)
    config_path = Path(output_dir) / "lightgbm_run_config.json"
    run_config = {
        "model": "lightgbm",
        "target": target,
        "window_months": window_months,
        "target_lags": target_lags,
        "test_start_date": (
            test_start.strftime("%Y-%m-%d") if test_start is not None else None
        ),
        "model_params": model_params,
        "params_file": str(Path(params_file).resolve()) if params_file else None,
    }
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2)
    paths["feature importance"] = importance_path
    paths["last-window model"] = model_path
    paths["run configuration"] = config_path

    print(
        f"Dataset: {Path(data_path).resolve()}\n"
        f"Target: {target}\n"
        f"Predictors: {len(features)} ({len(lag_columns)} target lag(s))\n"
        f"Rolling window: {window_months} months\n"
        f"Test start: {test_start.date() if test_start is not None else 'all'}\n"
        f"Out-of-sample months: {len(splits)}"
    )
    print_evaluation("ROLLING LIGHTGBM RESULTS", overall, by_country, paths)


if __name__ == "__main__":
    args = parse_args()
    main(
        args.data,
        args.output_dir,
        args.window_months,
        args.target_lags,
        args.n_estimators,
        args.params_file,
        args.test_start_date,
    )

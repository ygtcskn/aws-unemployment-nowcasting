"""Shared data and evaluation helpers for the forecasting models."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATE_COL = "date"
COUNTRY_COL = "country"
TARGET_COL = "unemployment_change_1m"
LEVEL_COL = "unemployment_rate"
LEVEL_LAG_COL = "unemployment_rate_lag1"


def load_tuned_config(path, expected_model, target):
    """Load an Optuna result and validate that it belongs to this model run."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tuned-parameter file not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    model = str(config.get("model", "")).lower().replace(" ", "")
    expected = expected_model.lower().replace(" ", "")
    if model != expected:
        raise ValueError(
            f"Parameter file is for {config.get('model')!r}, not {expected_model!r}."
        )
    if config.get("target") != target:
        raise ValueError(
            "Parameter file target does not match the modeling panel: "
            f"{config.get('target')!r} versus {target!r}."
        )

    params = config.get("best_params")
    if not isinstance(params, dict) or not params:
        raise ValueError("Parameter file does not contain a non-empty best_params.")

    test_start = config.get("recommended_test_start_date")
    if test_start is not None:
        test_start = pd.Timestamp(test_start)
        if pd.isna(test_start):
            raise ValueError("recommended_test_start_date is invalid.")

    return params.copy(), test_start, config


def exclude_target_lag_features(features, target):
    """Let a target-lag setting control precomputed lags in the input panel."""
    prefix = f"{target}_lag"
    return [
        column
        for column in features
        if not (
            column.startswith(prefix)
            and column.removeprefix(prefix).isdigit()
        )
    ]


def load_model_panel(path):
    """Load the change-target panel and guard against contemporaneous leakage."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Modeling panel not found: {path.resolve()}")

    data = pd.read_csv(path, parse_dates=[DATE_COL])
    if LEVEL_COL in data.columns:
        raise ValueError(
            f"The modeling panel contains {LEVEL_COL!r}. This contemporaneous "
            "unemployment rate would leak the outcome; retain only "
            f"{LEVEL_LAG_COL!r} as the level predictor."
        )

    required = {DATE_COL, COUNTRY_COL, TARGET_COL, LEVEL_LAG_COL}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Modeling panel is missing columns: {sorted(missing)}")

    target = TARGET_COL

    data = data.sort_values([COUNTRY_COL, DATE_COL]).reset_index(drop=True)
    if data[DATE_COL].isna().any():
        raise ValueError("The modeling panel contains invalid dates.")
    if data.duplicated([DATE_COL, COUNTRY_COL]).any():
        raise ValueError("The modeling panel contains duplicate country-months.")

    excluded = {DATE_COL, COUNTRY_COL, TARGET_COL}
    features = [column for column in data.columns if column not in excluded]
    if not features:
        raise ValueError("The modeling panel does not contain predictors.")

    nonnumeric = [
        column for column in features if not pd.api.types.is_numeric_dtype(data[column])
    ]
    if nonnumeric:
        raise ValueError(f"Predictors must be numeric: {nonnumeric[:10]}")

    values = data[features + [target]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("The modeling panel contains missing or non-finite values.")

    return data, target, features


def add_target_lags(data, target, lag_count):
    """Add unemployment lags within country and never bridge a calendar gap."""
    if lag_count < 0:
        raise ValueError("lag_count must be non-negative.")

    data = data.sort_values([COUNTRY_COL, DATE_COL]).copy()
    current_month = data[DATE_COL].dt.year * 12 + data[DATE_COL].dt.month
    grouped = data.groupby(COUNTRY_COL, sort=False)
    lag_columns = []

    for lag in range(1, lag_count + 1):
        lag_column = f"{target}_lag{lag}"
        previous_value = grouped[target].shift(lag)
        previous_date = grouped[DATE_COL].shift(lag)
        previous_month = previous_date.dt.year * 12 + previous_date.dt.month
        expected_lag = previous_value.where(
            current_month.sub(previous_month).eq(lag)
        )

        if lag_column in data.columns:
            comparable = expected_lag.notna()
            if not np.allclose(
                data.loc[comparable, lag_column],
                expected_lag.loc[comparable],
                equal_nan=True,
            ):
                raise ValueError(
                    f"Existing lag column {lag_column} is inconsistent with "
                    "the target history."
                )
        else:
            data[lag_column] = expected_lag
        lag_columns.append(lag_column)

    return data, lag_columns


def rolling_month_splits(data, window_months):
    """Return fixed-length rolling windows followed by one test month."""
    if window_months < 2:
        raise ValueError("window_months must be at least 2.")

    months = pd.Series(data[DATE_COL].drop_duplicates().sort_values().to_numpy())
    if len(months) <= window_months:
        raise ValueError(
            f"Only {len(months)} months are available; need more than "
            f"the {window_months}-month training window."
        )

    return [
        (months.iloc[index - window_months:index], months.iloc[index])
        for index in range(window_months, len(months))
    ]


def select_window(data, train_months, test_month):
    train = data[data[DATE_COL].isin(train_months)].copy()
    test = data[data[DATE_COL].eq(test_month)].copy()
    return train, test


def prediction_frame(test, actual, prediction, model_name, train_months):
    """Keep identifiers beside every genuinely out-of-sample prediction."""
    result = test[[DATE_COL, COUNTRY_COL]].copy()
    result["model"] = model_name
    result["actual"] = np.asarray(actual, dtype=float)
    result["prediction"] = np.asarray(prediction, dtype=float)
    result["train_start"] = train_months.iloc[0]
    result["train_end"] = train_months.iloc[-1]
    return result


def evaluate_predictions(predictions):
    """Calculate comparable pooled and country-level forecast errors."""
    predictions = predictions.copy()
    predictions["error"] = predictions["actual"] - predictions["prediction"]
    predictions["absolute_error"] = predictions["error"].abs()
    predictions["squared_error"] = predictions["error"].pow(2)

    overall_rows = []
    country_rows = []
    for model, group in predictions.groupby("model", sort=True):
        overall_rows.append(_metric_row(group, model=model))
        for country, country_group in group.groupby(COUNTRY_COL, sort=True):
            country_rows.append(
                _metric_row(country_group, model=model, country=country)
            )

    overall = pd.DataFrame(overall_rows).sort_values("RMSE").reset_index(drop=True)
    by_country = (
        pd.DataFrame(country_rows)
        .sort_values(["model", COUNTRY_COL])
        .reset_index(drop=True)
    )
    return predictions, overall, by_country


def _metric_row(data, model, country=None):
    row = {
        "model": model,
        "MAE": mean_absolute_error(data["actual"], data["prediction"]),
        "RMSE": np.sqrt(
            mean_squared_error(data["actual"], data["prediction"])
        ),
        "R2": (
            r2_score(data["actual"], data["prediction"])
            if len(data) >= 2
            else np.nan
        ),
        "N": len(data),
    }
    if country is not None:
        row[COUNTRY_COL] = country
    return row


def save_evaluation(predictions, overall, by_country, output_dir, prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "predictions": output_dir / f"{prefix}_predictions.csv",
        "overall metrics": output_dir / f"{prefix}_overall_metrics.csv",
        "country metrics": output_dir / f"{prefix}_country_metrics.csv",
    }
    predictions.to_csv(paths["predictions"], index=False)
    overall.to_csv(paths["overall metrics"], index=False)
    by_country.to_csv(paths["country metrics"], index=False)
    return paths


def print_evaluation(title, overall, by_country, paths):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    print("\nOverall out-of-sample performance:\n")
    print(overall.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nCountry-level performance:\n")
    print(by_country.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSaved files:")
    for label, path in paths.items():
        print(f"  {label}: {path.resolve()}")

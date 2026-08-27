"""Estimate rolling random-walk and pooled autoregressive benchmarks."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from model_utils import (
    evaluate_predictions,
    load_model_panel,
    prediction_frame,
    print_evaluation,
    rolling_month_splits,
    save_evaluation,
    select_window,
)


DATA_PATH = Path("data/final/gt_unemp_monthly_final.csv")
OUTPUT_DIR = Path("reports/outputs")
WINDOW_MONTHS = 60


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--window-months", type=int, default=WINDOW_MONTHS)
    parser.add_argument("--test-start-date")
    return parser.parse_args()


def main(data_path, output_dir, window_months, test_start_date=None):
    data, target, base_features = load_model_panel(data_path)
    country_dummies = [column for column in base_features if column.startswith("C_")]

    level_lag = "unemployment_rate_lag1"
    if level_lag not in data.columns:
        raise ValueError(
            f"The change target requires {level_lag} for the level AR(1)."
        )
    ar_features = [level_lag] + country_dummies

    splits = rolling_month_splits(data, window_months)
    test_start = None
    if test_start_date is not None:
        test_start = pd.Timestamp(test_start_date)
        if pd.isna(test_start):
            raise ValueError("test_start_date is invalid.")
        splits = [split for split in splits if pd.Timestamp(split[1]) >= test_start]
        if not splits:
            raise ValueError("No forecast months remain on or after test_start_date.")
    predictions = []

    for train_months, test_month in tqdm(
        splits, desc="Rolling baseline models", unit="month"
    ):
        train, test = select_window(data, train_months, test_month)

        rw_test = test[test[target].notna()].copy()
        rw_prediction = np.zeros(len(rw_test), dtype=float)

        if not rw_test.empty:
            predictions.append(
                prediction_frame(
                    rw_test,
                    rw_test[target],
                    rw_prediction,
                    "Random walk",
                    train_months,
                )
            )

        ar_train = train.dropna(subset=ar_features + [target]).copy()
        ar_test = test.dropna(subset=ar_features + [target]).copy()
        if not ar_train.empty and not ar_test.empty:
            model = LinearRegression()
            model.fit(ar_train[ar_features], ar_train[target])
            ar_prediction = model.predict(ar_test[ar_features])
            predictions.append(
                prediction_frame(
                    ar_test,
                    ar_test[target],
                    ar_prediction,
                    "AR(1)",
                    train_months,
                )
            )

    if not predictions:
        raise ValueError("No rolling baseline predictions were produced.")

    predictions = pd.concat(predictions, ignore_index=True)
    predictions, overall, by_country = evaluate_predictions(predictions)
    paths = save_evaluation(
        predictions, overall, by_country, output_dir, prefix="baseline"
    )

    print(
        f"Dataset: {Path(data_path).resolve()}\n"
        f"Target: {target}\n"
        f"Rolling window: {window_months} months\n"
        f"Test start: {test_start.date() if test_start is not None else 'all'}\n"
        f"Out-of-sample months: {len(splits)}"
    )
    print_evaluation("ROLLING BASELINE RESULTS", overall, by_country, paths)


if __name__ == "__main__":
    args = parse_args()
    main(
        args.data,
        args.output_dir,
        args.window_months,
        args.test_start_date,
    )

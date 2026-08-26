from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "data/final/gt_unemp_monthly_final.csv"
)

OUTPUT_DIR = Path(
    "data/results"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "baseline_predictions.csv"
)

COUNTRY_METRICS_FILE = (
    OUTPUT_DIR
    / "baseline_country_metrics.csv"
)

OVERALL_METRICS_FILE = (
    OUTPUT_DIR
    / "baseline_overall_metrics.csv"
)


DATE_COL = "date"
COUNTRY_COL = "country"
TARGET_COL = "unemployment_rate"

# Fixed rolling training window
WINDOW_MONTHS = 60

# AR(1) by default
# Change to 4 for AR(4), etc.
AR_LAGS = 1


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH,
    parse_dates=[DATE_COL],
)

df = (
    df
    .sort_values(
        [COUNTRY_COL, DATE_COL]
    )
    .reset_index(drop=True)
)


# ============================================================
# BASIC CHECKS
# ============================================================

required_columns = {
    DATE_COL,
    COUNTRY_COL,
    TARGET_COL,
}

missing_required = (
    required_columns
    - set(df.columns)
)

if missing_required:

    raise ValueError(
        f"Missing required columns: "
        f"{missing_required}"
    )


print(
    "\n========================================"
)

print(
    "POOLED ROLLING BASELINE MODELS"
)

print(
    "========================================"
)

print(
    f"\nDataset: {DATA_PATH}"
)

print(
    f"Rows: {len(df)}"
)

print(
    f"Countries: "
    f"{df[COUNTRY_COL].nunique()}"
)

print(
    f"Sample: "
    f"{df[DATE_COL].min().date()} "
    f"to "
    f"{df[DATE_COL].max().date()}"
)

print(
    f"Rolling window: "
    f"{WINDOW_MONTHS} months"
)

print(
    f"AR specification: "
    f"AR({AR_LAGS})"
)


# ============================================================
# CREATE UNEMPLOYMENT LAGS
#
# Lags are always constructed within country.
# ============================================================

for lag in range(
    1,
    AR_LAGS + 1
):

    df[
        f"unemployment_lag{lag}"
    ] = (
        df
        .groupby(COUNTRY_COL)[TARGET_COL]
        .shift(lag)
    )


LAG_COLS = [
    f"unemployment_lag{lag}"
    for lag in range(
        1,
        AR_LAGS + 1
    )
]


# ============================================================
# COUNTRY DUMMIES FOR POOLED AR
# ============================================================

country_dummies = pd.get_dummies(
    df[COUNTRY_COL],
    prefix="country",
    dtype=float,
)

df_model = pd.concat(
    [
        df,
        country_dummies
    ],
    axis=1,
)


COUNTRY_DUMMY_COLS = (
    country_dummies.columns.tolist()
)


AR_FEATURES = (
    LAG_COLS
    + COUNTRY_DUMMY_COLS
)


# ============================================================
# UNIQUE MONTHS
# ============================================================

months = (
    df_model[DATE_COL]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)


if len(months) <= WINDOW_MONTHS:

    raise ValueError(
        f"Only {len(months)} months available. "
        f"Need more than "
        f"{WINDOW_MONTHS} months."
    )


print(
    f"\nTotal months: "
    f"{len(months)}"
)

print(
    f"Potential out-of-sample months: "
    f"{len(months) - WINDOW_MONTHS}"
)


# ============================================================
# ROLLING WINDOW
# ============================================================

all_predictions = []


for test_idx in range(
    WINDOW_MONTHS,
    len(months)
):

    # --------------------------------------------------------
    # DEFINE TRAINING WINDOW + TEST MONTH
    # --------------------------------------------------------

    train_months = months.iloc[
        test_idx - WINDOW_MONTHS:
        test_idx
    ]

    test_month = months.iloc[
        test_idx
    ]

    train_start = (
        train_months.iloc[0]
    )

    train_end = (
        train_months.iloc[-1]
    )


    # --------------------------------------------------------
    # POOLED TRAINING SAMPLE
    # --------------------------------------------------------

    train = df_model[
        df_model[DATE_COL].isin(
            train_months
        )
    ].copy()


    test = df_model[
        df_model[DATE_COL]
        == test_month
    ].copy()


    if train.empty or test.empty:
        continue


    # ========================================================
    # RANDOM WALK / PERSISTENCE
    #
    # Forecast:
    #
    # u_hat(i,t) = u(i,t-1)
    # ========================================================

    rw_valid = (
        test[TARGET_COL].notna()
        & test["unemployment_lag1"].notna()
    )


    rw_test = test.loc[
        rw_valid
    ].copy()


    if not rw_test.empty:

        rw_result = pd.DataFrame(
            {
                "date": rw_test[
                    DATE_COL
                ].values,

                "country": rw_test[
                    COUNTRY_COL
                ].values,

                "model": "RW",

                "actual": rw_test[
                    TARGET_COL
                ].values,

                "prediction": rw_test[
                    "unemployment_lag1"
                ].values,

                "train_start": train_start,

                "train_end": train_end,
            }
        )


        all_predictions.append(
            rw_result
        )


    # ========================================================
    # POOLED AR MODEL
    # ========================================================

    train_valid = (
        train[
            AR_FEATURES
        ]
        .notna()
        .all(axis=1)
        & train[TARGET_COL].notna()
    )


    test_valid = (
        test[
            AR_FEATURES
        ]
        .notna()
        .all(axis=1)
        & test[TARGET_COL].notna()
    )


    ar_train = train.loc[
        train_valid
    ].copy()


    ar_test = test.loc[
        test_valid
    ].copy()


    if (
        not ar_train.empty
        and not ar_test.empty
    ):

        X_train = ar_train[
            AR_FEATURES
        ]

        y_train = ar_train[
            TARGET_COL
        ]


        X_test = ar_test[
            AR_FEATURES
        ]

        y_test = ar_test[
            TARGET_COL
        ]


        # ----------------------------------------------------
        # POOLED OLS AR MODEL
        # ----------------------------------------------------

        ar_model = LinearRegression()

        ar_model.fit(
            X_train,
            y_train,
        )


        ar_predictions = (
            ar_model.predict(
                X_test
            )
        )


        ar_result = pd.DataFrame(
            {
                "date": ar_test[
                    DATE_COL
                ].values,

                "country": ar_test[
                    COUNTRY_COL
                ].values,

                "model": f"AR({AR_LAGS})",

                "actual": y_test.values,

                "prediction": (
                    ar_predictions
                ),

                "train_start": train_start,

                "train_end": train_end,
            }
        )


        all_predictions.append(
            ar_result
        )


    print(
        f"Test: {test_month.date()} | "
        f"Train: {train_start.date()} "
        f"to {train_end.date()} | "
        f"N AR train: {len(ar_train)} | "
        f"N test: {len(ar_test)}"
    )


# ============================================================
# COMBINE ALL PREDICTIONS
# ============================================================

if not all_predictions:

    raise ValueError(
        "No rolling-window predictions "
        "were produced."
    )


predictions_df = pd.concat(
    all_predictions,
    ignore_index=True,
)


# ============================================================
# ERRORS
# ============================================================

predictions_df[
    "error"
] = (
    predictions_df["actual"]
    - predictions_df["prediction"]
)


predictions_df[
    "absolute_error"
] = (
    predictions_df["error"]
    .abs()
)


predictions_df[
    "squared_error"
] = (
    predictions_df["error"]
    ** 2
)


# ============================================================
# OVERALL MODEL PERFORMANCE
# ============================================================

overall_results = []


for model_name, group in (
    predictions_df.groupby(
        "model"
    )
):

    mae = mean_absolute_error(
        group["actual"],
        group["prediction"],
    )


    rmse = np.sqrt(
        mean_squared_error(
            group["actual"],
            group["prediction"],
        )
    )


    overall_results.append(
        {
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "N": len(group),
        }
    )


overall_results = pd.DataFrame(
    overall_results
)

overall_results = (
    overall_results
    .sort_values("RMSE")
    .reset_index(drop=True)
)


# ============================================================
# COUNTRY-LEVEL PERFORMANCE
# ============================================================

country_results = []


for (
    model_name,
    country
), group in predictions_df.groupby(
    [
        "model",
        "country"
    ]
):

    mae = mean_absolute_error(
        group["actual"],
        group["prediction"],
    )


    rmse = np.sqrt(
        mean_squared_error(
            group["actual"],
            group["prediction"],
        )
    )


    country_results.append(
        {
            "model": model_name,
            "country": country,
            "MAE": mae,
            "RMSE": rmse,
            "N": len(group),
        }
    )


country_results = pd.DataFrame(
    country_results
)

country_results = (
    country_results
    .sort_values(
        [
            "model",
            "RMSE"
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n========================================"
)

print(
    "OVERALL OUT-OF-SAMPLE RESULTS"
)

print(
    "========================================\n"
)


print(
    overall_results.to_string(
        index=False
    )
)


print(
    "\n========================================"
)

print(
    "COUNTRY-LEVEL RESULTS"
)

print(
    "========================================\n"
)


print(
    country_results.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


predictions_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


country_results.to_csv(
    COUNTRY_METRICS_FILE,
    index=False,
)


overall_results.to_csv(
    OVERALL_METRICS_FILE,
    index=False,
)


print(
    "\n========================================"
)

print(
    "FILES SAVED"
)

print(
    "========================================"
)


print(
    f"\nPredictions:"
    f"\n{OUTPUT_FILE}"
)


print(
    f"\nOverall metrics:"
    f"\n{OVERALL_METRICS_FILE}"
)


print(
    f"\nCountry metrics:"
    f"\n{COUNTRY_METRICS_FILE}"
)
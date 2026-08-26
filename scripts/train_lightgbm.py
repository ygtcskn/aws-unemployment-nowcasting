from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
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
    / "lightgbm_predictions.csv"
)

COUNTRY_METRICS_FILE = (
    OUTPUT_DIR
    / "lightgbm_country_metrics.csv"
)

DATE_COL = "date"
COUNTRY_COL = "country"
TARGET_COL = "unemployment_rate"

# Fixed rolling window:
# previous 60 months -> predict next month
WINDOW_MONTHS = 60

RANDOM_STATE = 42


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
        [DATE_COL, COUNTRY_COL]
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
    "POOLED LIGHTGBM ROLLING-WINDOW MODEL"
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


# ============================================================
# COUNTRY DUMMIES
# ============================================================

df = pd.get_dummies(
    df,
    columns=[COUNTRY_COL],
    prefix="country",
    dtype=float,
)

country_dummy_cols = [
    col
    for col in df.columns
    if col.startswith("country_")
]


# ============================================================
# DEFINE FEATURES
# ============================================================

exclude_cols = {
    DATE_COL,
    TARGET_COL,
}

FEATURES = [
    col
    for col in df.columns
    if col not in exclude_cols
]


print(
    f"\nNumber of predictors: "
    f"{len(FEATURES)}"
)


# ============================================================
# MODEL DATA
# ============================================================

model_df = df[
    [DATE_COL, TARGET_COL]
    + FEATURES
].copy()


# Target must be observed
model_df = model_df.dropna(
    subset=[TARGET_COL]
)


# ============================================================
# UNIQUE MONTHS
# ============================================================

months = (
    model_df[DATE_COL]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)


if len(months) <= WINDOW_MONTHS:
    raise ValueError(
        f"Only {len(months)} months available. "
        f"Need more than {WINDOW_MONTHS} months."
    )


print(
    f"Total months: {len(months)}"
)

print(
    f"Rolling window: "
    f"{WINDOW_MONTHS} months"
)

print(
    f"Out-of-sample months: "
    f"{len(months) - WINDOW_MONTHS}"
)


# ============================================================
# ROLLING WINDOW
# ============================================================

all_predictions = []


for test_idx in range(
    WINDOW_MONTHS,
    len(months),
):

    # --------------------------------------------------------
    # TRAIN / TEST MONTHS
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
    # POOLED TRAIN / TEST DATA
    # --------------------------------------------------------

    train = model_df[
        model_df[DATE_COL].isin(
            train_months
        )
    ].copy()

    test = model_df[
        model_df[DATE_COL]
        == test_month
    ].copy()


    if train.empty or test.empty:
        continue


    # --------------------------------------------------------
    # X / y
    # --------------------------------------------------------

    X_train = train[
        FEATURES
    ].copy()

    y_train = train[
        TARGET_COL
    ].copy()

    X_test = test[
        FEATURES
    ].copy()

    y_test = test[
        TARGET_COL
    ].copy()


    # --------------------------------------------------------
    # KEEP COMPLETE FEATURE ROWS
    #
    # LightGBM can technically handle NaNs, but this keeps
    # the setup consistent and explicit.
    # --------------------------------------------------------

    train_valid = (
        X_train
        .notna()
        .all(axis=1)
        & y_train.notna()
    )

    test_valid = (
        X_test
        .notna()
        .all(axis=1)
        & y_test.notna()
    )


    X_train = (
        X_train.loc[
            train_valid
        ]
    )

    y_train = (
        y_train.loc[
            train_valid
        ]
    )

    X_test = (
        X_test.loc[
            test_valid
        ]
    )

    y_test = (
        y_test.loc[
            test_valid
        ]
    )


    if X_train.empty:
        print(
            f"Skipping {test_month.date()}: "
            f"no valid training observations."
        )
        continue

    if X_test.empty:
        print(
            f"Skipping {test_month.date()}: "
            f"no valid test observations."
        )
        continue


    # ========================================================
    # LIGHTGBM
    # ========================================================

    model = LGBMRegressor(

        objective="regression",

        # Number of boosting trees
        n_estimators=500,

        # Shrinkage
        learning_rate=0.03,

        # Tree complexity
        num_leaves=31,
        max_depth=-1,

        # Minimum observations in leaves
        min_child_samples=20,

        # Feature / observation subsampling
        subsample=0.8,
        colsample_bytree=0.8,

        # Regularisation
        reg_alpha=0.1,
        reg_lambda=1.0,

        random_state=RANDOM_STATE,

        # Use all available CPU cores
        n_jobs=-1,

        # Reduce console output
        verbosity=-1,
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )


    # --------------------------------------------------------
    # PREDICT NEXT MONTH
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # RECOVER COUNTRY
    # --------------------------------------------------------

    test_country_dummies = (
        X_test[
            country_dummy_cols
        ]
    )

    country_names = (
        test_country_dummies
        .idxmax(axis=1)
        .str.replace(
            "country_",
            "",
            regex=False,
        )
    )


    # --------------------------------------------------------
    # SAVE CURRENT TEST MONTH
    # --------------------------------------------------------

    result = pd.DataFrame(
        {
            "date": (
                test.loc[
                    test_valid,
                    DATE_COL
                ].values
            ),
            "country": (
                country_names.values
            ),
            "actual": (
                y_test.values
            ),
            "prediction": (
                predictions
            ),
            "train_start": (
                train_start
            ),
            "train_end": (
                train_end
            ),
        }
    )


    all_predictions.append(
        result
    )


    print(
        f"Test: {test_month.date()} | "
        f"Train: {train_start.date()} "
        f"to {train_end.date()} | "
        f"N train: {len(X_train)} | "
        f"N test: {len(X_test)}"
    )


# ============================================================
# COMBINE ALL OUT-OF-SAMPLE PREDICTIONS
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
# OVERALL PERFORMANCE
# ============================================================

mae = mean_absolute_error(
    predictions_df["actual"],
    predictions_df["prediction"],
)

rmse = np.sqrt(
    mean_squared_error(
        predictions_df["actual"],
        predictions_df["prediction"],
    )
)


print(
    "\n========================================"
)

print(
    "LIGHTGBM OUT-OF-SAMPLE RESULTS"
)

print(
    "========================================"
)

print(
    f"\nMAE:  {mae:.4f}"
)

print(
    f"RMSE: {rmse:.4f}"
)

print(
    f"N predictions: "
    f"{len(predictions_df)}"
)


# ============================================================
# COUNTRY-LEVEL PERFORMANCE
# ============================================================

country_results = []


for country, group in (
    predictions_df.groupby(
        "country"
    )
):

    country_mae = (
        mean_absolute_error(
            group["actual"],
            group["prediction"],
        )
    )

    country_rmse = np.sqrt(
        mean_squared_error(
            group["actual"],
            group["prediction"],
        )
    )


    country_results.append(
        {
            "country": country,
            "MAE": country_mae,
            "RMSE": country_rmse,
            "N": len(group),
        }
    )


country_results = pd.DataFrame(
    country_results
)

country_results = (
    country_results
    .sort_values("RMSE")
    .reset_index(drop=True)
)


print(
    "\nCountry-level results:"
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
    f"\nCountry metrics:"
    f"\n{COUNTRY_METRICS_FILE}"
)
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/final/gt_unemp_monthly_final.csv")

OUTPUT_DIR = Path("output")
OUTPUT_FILE = OUTPUT_DIR / "xgboost_predictions.csv"

DATE_COL = "date"
COUNTRY_COL = "country"
TARGET_COL = "unemployment_rate"

# Fixed rolling-window length in months
WINDOW_MONTHS = 60

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH,
    parse_dates=[DATE_COL]
)

df = df.sort_values(
    [DATE_COL, COUNTRY_COL]
).reset_index(drop=True)


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


# ============================================================
# COUNTRY DUMMIES
# ============================================================

df = pd.get_dummies(
    df,
    columns=[COUNTRY_COL],
    prefix="country",
    dtype=float
)


# Keep country dummy names for later
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
    f"Number of features: {len(FEATURES)}"
)


# ============================================================
# REMOVE ROWS THAT CANNOT BE USED
# ============================================================

model_df = df[
    [DATE_COL, TARGET_COL] + FEATURES
].copy()

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


print(
    f"Sample: "
    f"{months.iloc[0].date()} "
    f"to {months.iloc[-1].date()}"
)

print(
    f"Total months: {len(months)}"
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
    # Define dates
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
    # Build train/test samples
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
    # Remove rows with missing predictors
    # --------------------------------------------------------

    train_valid = (
        X_train.notna().all(axis=1)
        & y_train.notna()
    )

    test_valid = (
        X_test.notna().all(axis=1)
        & y_test.notna()
    )


    X_train = X_train.loc[
        train_valid
    ]

    y_train = y_train.loc[
        train_valid
    ]

    X_test = X_test.loc[
        test_valid
    ]

    y_test = y_test.loc[
        test_valid
    ]


    if X_train.empty or X_test.empty:
        continue


    # --------------------------------------------------------
    # XGBOOST MODEL
    # --------------------------------------------------------

    model = XGBRegressor(
        objective="reg:squarederror",

        n_estimators=500,
        learning_rate=0.03,

        max_depth=4,

        min_child_weight=5,

        subsample=0.8,
        colsample_bytree=0.8,

        reg_alpha=0.1,
        reg_lambda=1.0,

        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    predictions = (
        model.predict(
            X_test
        )
    )


    # --------------------------------------------------------
    # SAVE MONTH RESULTS
    # --------------------------------------------------------

    result = pd.DataFrame(
        {
            "date": test.loc[
                test_valid,
                DATE_COL
            ].values,

            "actual": (
                y_test.values
            ),

            "prediction": (
                predictions
            ),

            "train_start": train_start,

            "train_end": train_end,
        }
    )


    # Recover country from dummy variables
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
            regex=False
        )
    )


    result["country"] = (
        country_names.values
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
# COMBINE PREDICTIONS
# ============================================================

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)


# ============================================================
# OVERALL PERFORMANCE
# ============================================================

mae = mean_absolute_error(
    predictions_df["actual"],
    predictions_df["prediction"]
)

rmse = np.sqrt(
    mean_squared_error(
        predictions_df["actual"],
        predictions_df["prediction"]
    )
)


print(
    "\n===================================="
)

print(
    "ROLLING XGBOOST RESULTS"
)

print(
    "===================================="
)

print(
    f"MAE:  {mae:.4f}"
)

print(
    f"RMSE: {rmse:.4f}"
)


# ============================================================
# COUNTRY PERFORMANCE
# ============================================================

country_results = (
    predictions_df
    .groupby("country")
    .apply(
        lambda x: pd.Series(
            {
                "MAE": mean_absolute_error(
                    x["actual"],
                    x["prediction"]
                ),

                "RMSE": np.sqrt(
                    mean_squared_error(
                        x["actual"],
                        x["prediction"]
                    )
                ),

                "N": len(x)
            }
        ),
        include_groups=False
    )
    .reset_index()
)


print(
    "\nCountry-level results:"
)

print(
    country_results
    .sort_values("RMSE")
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


predictions_df.to_csv(
    OUTPUT_FILE,
    index=False
)


country_results.to_csv(
    OUTPUT_DIR
    / "xgboost_country_metrics.csv",
    index=False
)


print(
    f"\nPredictions saved to:"
    f"\n{OUTPUT_FILE}"
)
"""
analysis.py

Reads model prediction CSV files from ./output and compares model RMSEs.

Outputs:
    output/analysis/model_rmse_summary.csv
    output/analysis/country_rmse.csv
    output/analysis/country_rmse_matrix.csv

If a date column is available, it also creates fair common-sample comparisons:
    output/analysis/model_rmse_summary_common_sample.csv
    output/analysis/country_rmse_common_sample.csv
    output/analysis/country_rmse_matrix_common_sample.csv

Expected prediction data can be either:

Long format:
    date, country, model, actual, prediction

or separate files per model:
    date, country, actual, prediction

or wide prediction format:
    date, country, actual, xgb_prediction, lasso_prediction, ...

The script tries to recognize common alternative column names automatically.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd()

OUTPUT_DIR = PROJECT_ROOT / "output"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Possible column names
# ---------------------------------------------------------------------

DATE_CANDIDATES = [
    "date",
    "month",
    "time",
    "period",
]

COUNTRY_CANDIDATES = [
    "country",
    "country_code",
    "iso2",
    "iso",
]

MODEL_CANDIDATES = [
    "model",
    "model_name",
    "method",
    "estimator",
]

ACTUAL_CANDIDATES = [
    "actual",
    "y_true",
    "true",
    "observed",
    "target",
    "unemployment",
    "y",
]

PREDICTION_CANDIDATES = [
    "prediction",
    "pred",
    "y_pred",
    "forecast",
    "nowcast",
    "predicted",
]

HORIZON_CANDIDATES = [
    "horizon",
    "h",
    "forecast_horizon",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def clean_colname(name: str) -> str:
    """Standardize a column name for matching."""
    name = str(name).strip().lower()
    name = re.sub(r"[\s\-]+", "_", name)
    return name


def first_existing(columns, candidates):
    """Return first candidate found in columns, otherwise None."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def infer_model_name_from_filename(path: Path) -> str:
    """
    Infer model name from filename when the CSV does not contain a model column.

    Examples:
        xgboost_predictions.csv -> xgboost
        lasso_rolling_predictions.csv -> lasso
    """
    name = path.stem.lower()

    removable = [
        "_rolling_predictions",
        "_predictions",
        "_prediction",
        "_forecast",
        "_forecasts",
        "_results",
        "_output",
        "_metrics",
    ]

    for ending in removable:
        if name.endswith(ending):
            name = name[: -len(ending)]

    return name.strip("_") or path.stem


def looks_like_prediction_column(col: str) -> bool:
    """Detect prediction columns in wide-format files."""
    patterns = (
        "_prediction",
        "_pred",
        "_forecast",
        "_nowcast",
        "prediction_",
        "pred_",
        "forecast_",
        "nowcast_",
    )
    return any(pattern in col for pattern in patterns)


def model_name_from_prediction_column(col: str) -> str:
    """Convert e.g. xgb_prediction -> xgb."""
    name = col

    prefixes = ["prediction_", "pred_", "forecast_", "nowcast_"]
    suffixes = ["_prediction", "_pred", "_forecast", "_nowcast"]

    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]

    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    return name.strip("_") or col


def load_prediction_file(path: Path) -> pd.DataFrame | None:
    """
    Read one CSV and convert it to a standardized long format:

        date | country | model | actual | prediction | horizon | source_file

    Returns None if the CSV does not look like prediction-level output.
    """
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[SKIP] Could not read {path.name}: {exc}")
        return None

    if df.empty:
        print(f"[SKIP] Empty CSV: {path.name}")
        return None

    df.columns = [clean_colname(c) for c in df.columns]
    cols = set(df.columns)

    date_col = first_existing(cols, DATE_CANDIDATES)
    country_col = first_existing(cols, COUNTRY_CANDIDATES)
    model_col = first_existing(cols, MODEL_CANDIDATES)
    actual_col = first_existing(cols, ACTUAL_CANDIDATES)
    pred_col = first_existing(cols, PREDICTION_CANDIDATES)
    horizon_col = first_existing(cols, HORIZON_CANDIDATES)

    # We need at least actual values.
    if actual_col is None:
        return None

    # --------------------------------------------------------------
    # Case 1: normal long format with one prediction column
    # --------------------------------------------------------------
    if pred_col is not None:
        out = pd.DataFrame()

        if date_col is not None:
            out["date"] = pd.to_datetime(df[date_col], errors="coerce")
        else:
            out["date"] = pd.NaT

        if country_col is not None:
            out["country"] = df[country_col].astype(str).str.upper().str.strip()
        else:
            out["country"] = "ALL"

        if model_col is not None:
            out["model"] = df[model_col].astype(str).str.strip()
        else:
            out["model"] = infer_model_name_from_filename(path)

        out["actual"] = pd.to_numeric(df[actual_col], errors="coerce")
        out["prediction"] = pd.to_numeric(df[pred_col], errors="coerce")

        if horizon_col is not None:
            out["horizon"] = df[horizon_col]
        else:
            out["horizon"] = np.nan

        out["source_file"] = path.name
        return out

    # --------------------------------------------------------------
    # Case 2: wide format with multiple model prediction columns
    # --------------------------------------------------------------
    prediction_columns = [
        c for c in df.columns
        if c != actual_col and looks_like_prediction_column(c)
    ]

    if not prediction_columns:
        return None

    pieces = []

    for pcol in prediction_columns:
        part = pd.DataFrame()

        if date_col is not None:
            part["date"] = pd.to_datetime(df[date_col], errors="coerce")
        else:
            part["date"] = pd.NaT

        if country_col is not None:
            part["country"] = df[country_col].astype(str).str.upper().str.strip()
        else:
            part["country"] = "ALL"

        part["model"] = model_name_from_prediction_column(pcol)
        part["actual"] = pd.to_numeric(df[actual_col], errors="coerce")
        part["prediction"] = pd.to_numeric(df[pcol], errors="coerce")

        if horizon_col is not None:
            part["horizon"] = df[horizon_col]
        else:
            part["horizon"] = np.nan

        part["source_file"] = path.name
        pieces.append(part)

    return pd.concat(pieces, ignore_index=True)


def rmse(actual: pd.Series, prediction: pd.Series) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((prediction - actual) ** 2)))


def prepare_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Clean concatenated predictions."""
    df = predictions.copy()

    df["model"] = df["model"].astype(str).str.strip()
    df["country"] = df["country"].astype(str).str.upper().str.strip()

    df = df[
        np.isfinite(df["actual"]) &
        np.isfinite(df["prediction"])
    ].copy()

    df["squared_error"] = (df["prediction"] - df["actual"]) ** 2

    return df


# ---------------------------------------------------------------------
# RMSE calculations
# ---------------------------------------------------------------------

def calculate_country_rmse(df: pd.DataFrame) -> pd.DataFrame:
    """Country-specific RMSE for every model."""
    result = (
        df.groupby(["model", "country"], as_index=False)
        .agg(
            rmse=("squared_error", lambda x: float(np.sqrt(x.mean()))),
            n_observations=("squared_error", "size"),
        )
    )

    result["country_rank"] = (
        result.groupby("country")["rmse"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    return result.sort_values(["country", "rmse", "model"])


def calculate_model_summary(
    df: pd.DataFrame,
    country_rmse: pd.DataFrame,
) -> pd.DataFrame:
    """
    Model-level comparison.

    pooled_rmse:
        RMSE calculated from every individual prediction together.

    average_country_rmse:
        First calculate RMSE separately for each country,
        then take the arithmetic mean across countries.
        Each country therefore gets equal weight.
    """
    pooled = (
        df.groupby("model", as_index=False)
        .agg(
            pooled_rmse=("squared_error", lambda x: float(np.sqrt(x.mean()))),
            n_observations=("squared_error", "size"),
            n_countries=("country", "nunique"),
        )
    )

    avg_country = (
        country_rmse.groupby("model", as_index=False)
        .agg(
            average_country_rmse=("rmse", "mean"),
            median_country_rmse=("rmse", "median"),
        )
    )

    summary = pooled.merge(avg_country, on="model", how="left")

    summary["pooled_rank"] = (
        summary["pooled_rmse"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    summary["average_country_rank"] = (
        summary["average_country_rmse"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    return summary.sort_values(["pooled_rmse", "average_country_rmse"])


def create_country_matrix(country_rmse: pd.DataFrame) -> pd.DataFrame:
    """Country x model table for easy comparison."""
    matrix = country_rmse.pivot(
        index="country",
        columns="model",
        values="rmse",
    )

    matrix["best_model"] = matrix.idxmin(axis=1)
    matrix["best_rmse"] = matrix.drop(columns=["best_model"]).min(axis=1)

    return matrix.reset_index()


# ---------------------------------------------------------------------
# Common-sample comparison
# ---------------------------------------------------------------------

def restrict_to_common_sample(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Restrict evaluation to country-date observations available for every model.

    This prevents a model from looking better simply because it was evaluated
    on an easier or smaller set of observations.

    Requires a valid date column.
    """
    usable = df[df["date"].notna()].copy()

    if usable.empty:
        return None

    models = sorted(usable["model"].unique())

    if len(models) < 2:
        return None

    usable["obs_key"] = (
        usable["country"].astype(str)
        + "||"
        + usable["date"].dt.strftime("%Y-%m-%d")
    )

    model_keys = {
        model: set(group["obs_key"])
        for model, group in usable.groupby("model")
    }

    common_keys = set.intersection(*model_keys.values())

    if not common_keys:
        return None

    common = usable[usable["obs_key"].isin(common_keys)].copy()

    # If a model accidentally contains duplicate predictions for the same
    # country/date, keep only one and warn.
    duplicate_mask = common.duplicated(
        subset=["model", "country", "date"],
        keep=False,
    )

    if duplicate_mask.any():
        n_dup = int(duplicate_mask.sum())
        print(
            f"[WARNING] Found {n_dup} duplicate model-country-date rows "
            "in the common sample. Keeping the last occurrence."
        )
        common = common.drop_duplicates(
            subset=["model", "country", "date"],
            keep="last",
        )

    common.drop(columns=["obs_key"], inplace=True)

    return common


# ---------------------------------------------------------------------
# Optional horizon-specific results
# ---------------------------------------------------------------------

def calculate_horizon_summary(df: pd.DataFrame) -> pd.DataFrame | None:
    """Calculate pooled RMSE by model and horizon if horizon exists."""
    if "horizon" not in df.columns or df["horizon"].isna().all():
        return None

    result = (
        df.dropna(subset=["horizon"])
        .groupby(["model", "horizon"], as_index=False)
        .agg(
            rmse=("squared_error", lambda x: float(np.sqrt(x.mean()))),
            n_observations=("squared_error", "size"),
            n_countries=("country", "nunique"),
        )
        .sort_values(["horizon", "rmse"])
    )

    return result


# ---------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------

def print_summary(title: str, summary: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    display_cols = [
        "model",
        "pooled_rmse",
        "average_country_rmse",
        "median_country_rmse",
        "n_countries",
        "n_observations",
        "pooled_rank",
        "average_country_rank",
    ]

    print(
        summary[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    if not OUTPUT_DIR.exists():
        raise FileNotFoundError(
            f"Output folder not found:\n{OUTPUT_DIR}\n\n"
            "Create ./output or change OUTPUT_DIR in analysis.py."
        )

    csv_files = sorted(
        p for p in OUTPUT_DIR.rglob("*.csv")
        if ANALYSIS_DIR not in p.parents
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found under:\n{OUTPUT_DIR}"
        )

    print(f"Searching prediction outputs in: {OUTPUT_DIR}")
    print(f"Found {len(csv_files)} CSV file(s).")

    pieces = []

    for path in csv_files:
        loaded = load_prediction_file(path)

        if loaded is None:
            print(f"[SKIP] Not a prediction-level CSV: {path.relative_to(OUTPUT_DIR)}")
            continue

        print(
            f"[LOAD] {path.relative_to(OUTPUT_DIR)} "
            f"-> {len(loaded):,} prediction row(s)"
        )
        pieces.append(loaded)

    if not pieces:
        raise ValueError(
            "No prediction-level CSV could be identified.\n\n"
            "The script expects columns such as:\n"
            "  date, country, model, actual, prediction\n"
            "or:\n"
            "  date, country, actual, prediction\n"
            "or wide prediction columns such as xgb_prediction."
        )

    predictions = pd.concat(pieces, ignore_index=True)
    predictions = prepare_predictions(predictions)

    if predictions.empty:
        raise ValueError("No valid actual/prediction pairs remain after cleaning.")

    print(f"\nValid predictions loaded: {len(predictions):,}")
    print(f"Models: {sorted(predictions['model'].unique())}")
    print(f"Countries: {sorted(predictions['country'].unique())}")

    # --------------------------------------------------------------
    # Full available sample
    # --------------------------------------------------------------
    country_rmse = calculate_country_rmse(predictions)
    model_summary = calculate_model_summary(predictions, country_rmse)
    country_matrix = create_country_matrix(country_rmse)

    model_summary.to_csv(
        ANALYSIS_DIR / "model_rmse_summary.csv",
        index=False,
    )

    country_rmse.to_csv(
        ANALYSIS_DIR / "country_rmse.csv",
        index=False,
    )

    country_matrix.to_csv(
        ANALYSIS_DIR / "country_rmse_matrix.csv",
        index=False,
    )

    print_summary("MODEL RMSE COMPARISON — ALL AVAILABLE OBSERVATIONS", model_summary)

    print("\nCountry-specific RMSE:")
    print(
        country_rmse.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # --------------------------------------------------------------
    # Common sample
    # --------------------------------------------------------------
    common = restrict_to_common_sample(predictions)

    if common is not None and not common.empty:
        common_country_rmse = calculate_country_rmse(common)
        common_summary = calculate_model_summary(common, common_country_rmse)
        common_matrix = create_country_matrix(common_country_rmse)

        common_summary.to_csv(
            ANALYSIS_DIR / "model_rmse_summary_common_sample.csv",
            index=False,
        )

        common_country_rmse.to_csv(
            ANALYSIS_DIR / "country_rmse_common_sample.csv",
            index=False,
        )

        common_matrix.to_csv(
            ANALYSIS_DIR / "country_rmse_matrix_common_sample.csv",
            index=False,
        )

        print_summary(
            "MODEL RMSE COMPARISON — COMMON COUNTRY-DATE SAMPLE",
            common_summary,
        )

        print(
            "\nFor direct model comparison, the COMMON-SAMPLE table is the "
            "preferred table when models have different observation counts."
        )
    else:
        print(
            "\n[INFO] No common-sample table created. "
            "A usable date column for at least two models is required."
        )

    # --------------------------------------------------------------
    # Horizon-specific results, if applicable
    # --------------------------------------------------------------
    horizon_summary = calculate_horizon_summary(predictions)

    if horizon_summary is not None:
        horizon_summary.to_csv(
            ANALYSIS_DIR / "horizon_rmse.csv",
            index=False,
        )

        print("\nHorizon-specific RMSE:")
        print(
            horizon_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

    print("\nSaved analysis files to:")
    print(ANALYSIS_DIR)


if __name__ == "__main__":
    main()

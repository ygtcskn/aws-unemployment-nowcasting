"""Tune XGBoost with rolling validation and a protected final holdout.

The Optuna study only sees validation months that end before the recommended
test period.  Hyperparameters can therefore be selected without using the
months that will later be reported as the final out-of-sample evaluation.
"""

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from xgboost import XGBRegressor

from model_utils import (
    COUNTRY_COL,
    DATE_COL,
    add_target_lags,
    exclude_target_lag_features,
    load_model_panel,
    rolling_month_splits,
    select_window,
)


DATA_PATH = Path("data/final/gt_unemp_monthly_final.csv")
OUTPUT_PATH = Path("reports/outputs/tuning/xgboost_optuna_best.json")
DEFAULT_HOLDOUT_MONTHS = 24
DEFAULT_WINDOW_OPTIONS = (60, 84, 120)
DEFAULT_LAG_OPTIONS = (0, 1, 2, 3)
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--validation-months", type=int, default=24)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Optional wall-clock limit for the complete Optuna study.",
    )
    parser.add_argument(
        "--model-jobs",
        type=int,
        default=-1,
        help="Threads used by each XGBoost fit. Optuna trials stay sequential.",
    )
    parser.add_argument(
        "--window-options",
        type=int,
        nargs="+",
        default=list(DEFAULT_WINDOW_OPTIONS),
    )
    parser.add_argument(
        "--lag-options",
        type=int,
        nargs="+",
        default=list(DEFAULT_LAG_OPTIONS),
    )
    cutoff = parser.add_mutually_exclusive_group()
    cutoff.add_argument(
        "--tuning-end",
        help="Last month Optuna may evaluate, written as YYYY-MM.",
    )
    cutoff.add_argument(
        "--holdout-months",
        type=int,
        help=(
            "Number of final months hidden from Optuna. The default is "
            f"{DEFAULT_HOLDOUT_MONTHS} when no tuning end is supplied."
        ),
    )
    parser.add_argument("--pruning-startup-trials", type=int, default=5)
    parser.add_argument("--pruning-warmup-months", type=int, default=3)
    parser.add_argument("--study-name", default="xgboost_rolling_tuning")
    parser.add_argument(
        "--storage",
        help="Optional Optuna storage URL, for example sqlite:///xgb.db.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing named study; requires --storage.",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run one trial on two validation months. This checks the full "
            "pipeline but is not a meaningful tuning exercise."
        ),
    )
    return parser.parse_args()


def _unique_positive(values, label, minimum=1):
    values = list(dict.fromkeys(values))
    if not values or any(value < minimum for value in values):
        raise ValueError(f"{label} must contain integers of at least {minimum}.")
    return values


def _month_end(value):
    try:
        return pd.Period(value, freq="M").to_timestamp("M")
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid tuning end {value!r}; use the YYYY-MM format."
        ) from error


def define_validation_period(
    data,
    max_window,
    validation_months,
    tuning_end=None,
    holdout_months=None,
):
    """Choose common validation dates and keep all later dates untouched."""
    months = pd.DatetimeIndex(data[DATE_COL].drop_duplicates().sort_values())
    if validation_months < 1:
        raise ValueError("validation_months must be positive.")

    if tuning_end is None:
        reserved = (
            DEFAULT_HOLDOUT_MONTHS
            if holdout_months is None
            else holdout_months
        )
        if reserved < 1:
            raise ValueError("holdout_months must be at least 1.")
        if reserved >= len(months):
            raise ValueError("The holdout would use the complete dataset.")
        cutoff_index = len(months) - reserved - 1
        requested_cutoff = months[cutoff_index]
    else:
        requested_cutoff = _month_end(tuning_end)
        eligible = np.flatnonzero(months <= requested_cutoff)
        if len(eligible) == 0:
            raise ValueError("The tuning end is before the first panel month.")
        cutoff_index = int(eligible[-1])
        reserved = len(months) - cutoff_index - 1
        if reserved < 1:
            raise ValueError(
                "The tuning end must leave at least one later month as holdout."
            )

    earliest_index = max_window
    first_validation_index = cutoff_index - validation_months + 1
    if first_validation_index < earliest_index:
        available = max(0, cutoff_index - earliest_index + 1)
        raise ValueError(
            f"Only {available} common validation months remain after allowing "
            f"the largest {max_window}-month training window."
        )

    validation = months[first_validation_index : cutoff_index + 1]
    holdout = months[cutoff_index + 1 :]
    return {
        "validation_months": validation,
        "holdout_months": holdout,
        "resolved_tuning_end": months[cutoff_index],
        "requested_tuning_end": requested_cutoff,
        "recommended_test_start": holdout[0],
    }


def suggest_parameters(trial, window_options, lag_options, smoke_test=False):
    """Use a conservative search space for a short, high-dimensional panel."""
    if smoke_test:
        n_estimators = trial.suggest_int("n_estimators", 20, 40, step=20)
    else:
        n_estimators = trial.suggest_int(
            "n_estimators", 100, 800, step=50
        )

    return {
        "window_months": trial.suggest_categorical(
            "window_months", window_options
        ),
        "target_lags": trial.suggest_categorical(
            "target_lags", lag_options
        ),
        "n_estimators": n_estimators,
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.005, 0.08, log=True
        ),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 5.0, 80.0, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.2, 0.8
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-4, 10.0, log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 0.5, 50.0, log=True
        ),
        "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
    }


class RollingObjective:
    """Evaluate every trial on identical, chronologically ordered months."""

    def __init__(
        self,
        data,
        target,
        base_features,
        validation_months,
        window_options,
        lag_options,
        seed,
        model_jobs,
        smoke_test,
    ):
        self.data = data
        self.target = target
        self.base_features = base_features
        self.validation_months = pd.DatetimeIndex(validation_months)
        self.validation_set = set(self.validation_months)
        self.window_options = window_options
        self.lag_options = lag_options
        self.seed = seed
        self.model_jobs = model_jobs
        self.smoke_test = smoke_test
        self.lag_cache = {}

    def _prepared_data(self, lag_count):
        if lag_count not in self.lag_cache:
            prepared, lag_columns = add_target_lags(
                self.data, self.target, lag_count
            )
            features = list(dict.fromkeys(self.base_features + lag_columns))
            self.lag_cache[lag_count] = (prepared, features)
        return self.lag_cache[lag_count]

    def __call__(self, trial):
        params = suggest_parameters(
            trial,
            self.window_options,
            self.lag_options,
            self.smoke_test,
        )
        window_months = params.pop("window_months")
        target_lags = params.pop("target_lags")
        prepared, features = self._prepared_data(target_lags)
        splits = [
            split
            for split in rolling_month_splits(prepared, window_months)
            if split[1] in self.validation_set
        ]

        if len(splits) != len(self.validation_months):
            raise RuntimeError(
                "A trial could not construct every common validation month."
            )

        squared_error_sum = 0.0
        prediction_count = 0
        for step, (train_months, test_month) in enumerate(splits):
            train, test = select_window(prepared, train_months, test_month)
            train = train.dropna(subset=features + [self.target])
            test = test.dropna(subset=features + [self.target])
            if train.empty or test.empty:
                raise RuntimeError(
                    f"Empty training or validation sample at {test_month:%Y-%m}."
                )

            model = XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                eval_metric="rmse",
                random_state=self.seed,
                n_jobs=self.model_jobs,
                verbosity=0,
                **params,
            )
            model.fit(train[features], train[self.target])
            prediction = model.predict(test[features])
            errors = test[self.target].to_numpy(dtype=float) - prediction
            squared_error_sum += float(np.dot(errors, errors))
            prediction_count += len(errors)

            running_rmse = np.sqrt(squared_error_sum / prediction_count)
            trial.report(float(running_rmse), step=step)
            if trial.should_prune():
                trial.set_user_attr("months_evaluated", step + 1)
                trial.set_user_attr("predictions_evaluated", prediction_count)
                raise optuna.TrialPruned()

        trial.set_user_attr("months_evaluated", len(splits))
        trial.set_user_attr("predictions_evaluated", prediction_count)
        trial.set_user_attr("feature_count", len(features))
        return float(np.sqrt(squared_error_sum / prediction_count))


def validation_random_walk_rmse(data, target, validation_months):
    sample = data[data[DATE_COL].isin(validation_months)].copy()
    forecast = np.zeros(len(sample), dtype=float)

    return float(
        np.sqrt(mean_squared_error(sample[target], forecast))
    )


def _iso_month(value):
    return pd.Timestamp(value).strftime("%Y-%m")


def save_results(
    study,
    output_path,
    data_path,
    data,
    target,
    base_features,
    period,
    args,
    window_options,
    lag_options,
    rw_rmse,
):
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        raise RuntimeError("Optuna did not complete any trials.")

    best = study.best_trial
    best_params = dict(best.params)
    model_params = {
        key: value
        for key, value in best_params.items()
        if key not in {"window_months", "target_lags"}
    }
    result = {
        "model": "xgboost",
        "best_value_rmse": float(best.value),
        "tuning_end_date": _iso_month(period["resolved_tuning_end"]),
        "recommended_test_start_date": _iso_month(
            period["recommended_test_start"]
        ),
        "validation_months": len(period["validation_months"]),
        "reserved_holdout_months": len(period["holdout_months"]),
        "n_trials": len(study.trials),
        "seed": args.seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(Path(data_path).resolve()),
        "target": target,
        "objective": "pooled rolling-validation RMSE",
        "best_trial": best.number,
        "best_validation_rmse": float(best.value),
        "validation_random_walk_rmse": rw_rmse,
        "rmse_ratio_to_random_walk": float(best.value / rw_rmse),
        "best_params": best_params,
        "training_window_months": best_params["window_months"],
        "target_lags": best_params["target_lags"],
        "xgboost_params": model_params,
        "validation": {
            "start": _iso_month(period["validation_months"][0]),
            "end": _iso_month(period["validation_months"][-1]),
            "months": len(period["validation_months"]),
            "predictions_in_best_trial": best.user_attrs.get(
                "predictions_evaluated"
            ),
        },
        "protected_holdout": {
            "start": _iso_month(period["recommended_test_start"]),
            "end": _iso_month(period["holdout_months"][-1]),
            "months": len(period["holdout_months"]),
            "used_by_optuna": False,
        },
        "recommended_test_start": _iso_month(
            period["recommended_test_start"]
        ),
        "panel": {
            "start": _iso_month(data[DATE_COL].min()),
            "end": _iso_month(data[DATE_COL].max()),
            "months": int(data[DATE_COL].nunique()),
            "countries": sorted(data[COUNTRY_COL].unique().tolist()),
            "rows": len(data),
            "base_features": len(base_features),
        },
        "study": {
            "sampler": "TPESampler",
            "seed": args.seed,
            "pruner": "MedianPruner",
            "pruning_startup_trials": args.pruning_startup_trials,
            "pruning_warmup_months": args.pruning_warmup_months,
            "requested_trials": args.n_trials,
            "completed_trials": len(completed),
            "pruned_trials": sum(
                trial.state == optuna.trial.TrialState.PRUNED
                for trial in study.trials
            ),
            "window_options": window_options,
            "target_lag_options": lag_options,
            "smoke_test": args.smoke_test,
            "study_name": args.study_name,
            "storage": args.storage,
        },
        "versions": {
            "python": platform.python_version(),
            "optuna": optuna.__version__,
            "xgboost": xgboost.__version__,
        },
        "protocol_note": (
            "Only the development-period rolling validation months were used "
            "for selection. Start final evaluation at recommended_test_start "
            "and do not tune again on that protected holdout."
        ),
        "upstream_preprocessing_note": (
            "This script protects the target holdout. Strict real-time claims "
            "also require every upstream feature transformation to be causal "
            "or re-estimated using training data only."
        ),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    history_path = output_path.with_name(
        f"{output_path.stem}_trials.csv"
    )
    study.trials_dataframe().to_csv(history_path, index=False)
    return result, history_path


def main(args):
    if args.resume and not args.storage:
        raise ValueError("--resume requires --storage.")
    if args.smoke_test:
        args.n_trials = 1
        args.validation_months = 2
    if args.n_trials < 1:
        raise ValueError("n_trials must be positive.")
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive when supplied.")
    if args.model_jobs == 0:
        raise ValueError("model_jobs must be -1 or a positive integer.")
    if args.pruning_startup_trials < 0 or args.pruning_warmup_months < 0:
        raise ValueError("Pruning settings cannot be negative.")

    window_options = _unique_positive(
        args.window_options, "window_options", minimum=2
    )
    lag_options = _unique_positive(
        args.lag_options, "lag_options", minimum=0
    )
    data, target, base_features = load_model_panel(args.data)
    base_features = exclude_target_lag_features(base_features, target)
    period = define_validation_period(
        data=data,
        max_window=max(window_options),
        validation_months=args.validation_months,
        tuning_end=args.tuning_end,
        holdout_months=args.holdout_months,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.pruning_startup_trials,
        n_warmup_steps=max(args.pruning_warmup_months - 1, 0),
        interval_steps=1,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=args.resume,
    )
    objective = RollingObjective(
        data=data,
        target=target,
        base_features=base_features,
        validation_months=period["validation_months"],
        window_options=window_options,
        lag_options=lag_options,
        seed=args.seed,
        model_jobs=args.model_jobs,
        smoke_test=args.smoke_test,
    )

    progress = tqdm(
        total=args.n_trials,
        desc="Optuna XGBoost",
        unit="trial",
        disable=args.no_progress,
    )

    def update_progress(study, trial):
        progress.update(1)
        completed = [
            item
            for item in study.trials
            if item.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed and study.best_trial.number == trial.number:
            progress.set_postfix(best_rmse=f"{study.best_value:.4f}")

    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout_seconds,
            callbacks=[update_progress],
            gc_after_trial=True,
        )
    finally:
        progress.close()

    rw_rmse = validation_random_walk_rmse(
        data, target, period["validation_months"]
    )
    result, history_path = save_results(
        study=study,
        output_path=args.output,
        data_path=args.data,
        data=data,
        target=target,
        base_features=base_features,
        period=period,
        args=args,
        window_options=window_options,
        lag_options=lag_options,
        rw_rmse=rw_rmse,
    )

    print("\nXGBOOST OPTUNA SEARCH COMPLETE")
    print(f"Target: {target}")
    print(
        "Validation: "
        f"{result['validation']['start']} to {result['validation']['end']}"
    )
    print(
        "Protected test period: "
        f"{result['protected_holdout']['start']} to "
        f"{result['protected_holdout']['end']}"
    )
    print(f"Best validation RMSE: {result['best_validation_rmse']:.4f}")
    print(
        "Validation RMSE / random-walk RMSE: "
        f"{result['rmse_ratio_to_random_walk']:.4f}"
    )
    print(f"Best settings: {result['best_params']}")
    print(f"Saved result: {Path(args.output).resolve()}")
    print(f"Saved trial history: {history_path.resolve()}")


if __name__ == "__main__":
    main(parse_args())

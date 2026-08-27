"""Tune LightGBM with rolling validation and an untouched final holdout."""

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError as exc:
    raise SystemExit(
        "Optuna is required for tuning. Install the project requirements or "
        "run: pip install optuna"
    ) from exc

from lightgbm import LGBMRegressor

from model_utils import (
    DATE_COL,
    add_target_lags,
    exclude_target_lag_features,
    load_model_panel,
    select_window,
)


DATA_PATH = Path("data/final/gt_unemp_monthly_final.csv")
OUTPUT_DIR = Path("reports/outputs/tuning")
DEFAULT_HOLDOUT_MONTHS = 24
DEFAULT_WINDOW_CHOICES = (60, 84, 120)
DEFAULT_TARGET_LAG_CHOICES = (0, 1, 2, 3)
RANDOM_STATE = 42


def parse_month(value):
    """Convert a YYYY-MM argument to a monthly period with a clear error."""
    try:
        period = pd.Period(value, freq="M")
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"Expected YYYY-MM, received {value!r}."
        ) from exc
    return period


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("The value must be positive.")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("The value cannot be negative.")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Quick smoke test: python scripts/tune_lightgbm.py --smoke-test"
        ),
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-trials", type=positive_int, default=50)
    parser.add_argument("--validation-months", type=positive_int, default=24)
    parser.add_argument("--timeout-seconds", type=positive_int)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--window-choices",
        type=positive_int,
        nargs="+",
        default=list(DEFAULT_WINDOW_CHOICES),
    )
    parser.add_argument(
        "--target-lag-choices",
        type=nonnegative_int,
        nargs="+",
        default=list(DEFAULT_TARGET_LAG_CHOICES),
    )
    parser.add_argument(
        "--model-jobs",
        type=int,
        default=-1,
        help="Threads used by each LightGBM fit.",
    )
    parser.add_argument(
        "--pruner-startup-trials",
        type=nonnegative_int,
        default=5,
    )
    parser.add_argument(
        "--pruner-warmup-months",
        type=nonnegative_int,
        default=3,
    )
    parser.add_argument(
        "--study-name",
        default="lightgbm_rolling_tuning",
    )
    parser.add_argument(
        "--storage",
        help="Optional Optuna storage URL, for example sqlite:///lgbm.db.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing named study; requires --storage.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Optuna's trial progress bar.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run one trial over two validation months with at most 50 trees. "
            "Outputs are written below an isolated smoke directory."
        ),
    )

    boundary = parser.add_mutually_exclusive_group()
    boundary.add_argument(
        "--tuning-end",
        type=parse_month,
        help="Last development month in YYYY-MM form.",
    )
    boundary.add_argument(
        "--holdout-months",
        type=positive_int,
        help=(
            "Number of final months excluded from tuning. If neither boundary "
            f"option is given, {DEFAULT_HOLDOUT_MONTHS} months are reserved."
        ),
    )
    return parser.parse_args()


def package_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def unique_sorted(values):
    return sorted(set(values))


def monthly_axis(data):
    months = pd.DatetimeIndex(data[DATE_COL].drop_duplicates().sort_values())
    periods = months.to_period("M")
    if periods.duplicated().any():
        raise ValueError("The panel contains more than one date in a month.")
    if len(periods) > 1 and not np.all(np.diff(periods.asi8) == 1):
        raise ValueError(
            "The panel has a gap in its monthly date axis. Rolling validation "
            "requires consecutive months."
        )
    return months, periods


def development_boundary(months, periods, tuning_end, holdout_months):
    """Separate development and holdout months without inspecting outcomes."""
    if tuning_end is not None:
        matches = np.flatnonzero(periods == tuning_end)
        if len(matches) != 1:
            raise ValueError(
                f"The requested tuning end {tuning_end} is not in the panel."
            )
        development_end_index = int(matches[0])
    else:
        holdout_months = (
            DEFAULT_HOLDOUT_MONTHS
            if holdout_months is None
            else holdout_months
        )
        if holdout_months >= len(months):
            raise ValueError(
                f"Cannot reserve {holdout_months} of the {len(months)} months."
            )
        development_end_index = len(months) - holdout_months - 1

    development = months[: development_end_index + 1]
    holdout = months[development_end_index + 1 :]
    if len(holdout) < 1:
        raise ValueError(
            "Tuning must leave at least one final month as a protected holdout."
        )
    return development, holdout


def validation_design(
    data,
    tuning_end,
    holdout_months,
    validation_months,
    requested_windows,
):
    months, periods = monthly_axis(data)
    development, holdout = development_boundary(
        months, periods, tuning_end, holdout_months
    )
    if validation_months >= len(development):
        raise ValueError(
            f"Cannot use {validation_months} validation months with only "
            f"{len(development)} development months."
        )

    validation = development[-validation_months:]
    months_before_first_validation = len(development) - validation_months
    requested_windows = unique_sorted(requested_windows)
    feasible_windows = [
        window
        for window in requested_windows
        if window <= months_before_first_validation
    ]
    if not feasible_windows:
        raise ValueError(
            "None of the requested training windows fits before the first "
            f"validation month. At most {months_before_first_validation} "
            "months are feasible."
        )

    month_positions = {month: index for index, month in enumerate(months)}
    return {
        "all_months": months,
        "development_months": development,
        "holdout_months": holdout,
        "validation_months": validation,
        "month_positions": month_positions,
        "requested_windows": requested_windows,
        "feasible_windows": feasible_windows,
    }


def suggest_model_params(trial, smoke_test=False):
    """Search conservative trees and strong regularization for a small panel."""
    estimator_high = 50 if smoke_test else 800
    estimator_low = 20 if smoke_test else 100
    estimator_step = 10 if smoke_test else 50
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            estimator_low,
            estimator_high,
            step=estimator_step,
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.005, 0.08, log=True
        ),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "num_leaves": trial.suggest_categorical(
            "num_leaves", [3, 5, 7, 11, 15, 23, 31]
        ),
        "min_child_samples": trial.suggest_int(
            "min_child_samples", 20, 120, step=10
        ),
        "min_split_gain": trial.suggest_float(
            "min_split_gain", 0.0, 0.1
        ),
        "subsample": trial.suggest_float(
            "subsample", 0.6, 1.0, step=0.05
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.15, 0.8, step=0.05
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-4, 10.0, log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 0.1, 100.0, log=True
        ),
        "max_bin": trial.suggest_categorical("max_bin", [31, 63, 127]),
    }


def make_model(params, seed, model_jobs):
    return LGBMRegressor(
        objective="regression",
        subsample_freq=1,
        random_state=seed,
        n_jobs=model_jobs,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        **params,
    )


class RollingObjective:
    """Evaluate each trial on the same chronological development forecasts."""

    def __init__(
        self,
        data,
        target,
        base_features,
        design,
        lag_choices,
        seed,
        model_jobs,
        smoke_test,
    ):
        self.data = data
        self.target = target
        self.base_features = base_features
        self.all_months = design["all_months"]
        self.validation_months = design["validation_months"]
        self.month_positions = design["month_positions"]
        self.window_choices = design["feasible_windows"]
        self.lag_choices = lag_choices
        self.seed = seed
        self.model_jobs = model_jobs
        self.smoke_test = smoke_test

    def __call__(self, trial):
        window_months = trial.suggest_categorical(
            "window_months", self.window_choices
        )
        target_lags = trial.suggest_categorical(
            "target_lags", self.lag_choices
        )
        model_params = suggest_model_params(trial, self.smoke_test)
        trial_data, lag_columns = add_target_lags(
            self.data, self.target, target_lags
        )
        features = list(dict.fromkeys(self.base_features + lag_columns))

        squared_error_sum = 0.0
        observation_count = 0
        monthly_rmse = []

        for step, test_month in enumerate(self.validation_months):
            test_index = self.month_positions[test_month]
            train_months = pd.Series(
                self.all_months[
                    test_index - window_months : test_index
                ].to_numpy()
            )
            if len(train_months) != window_months:
                raise RuntimeError("The selected rolling window is incomplete.")

            train, test = select_window(trial_data, train_months, test_month)
            train = train.dropna(subset=features + [self.target])
            test = test.dropna(subset=features + [self.target])
            if train.empty or test.empty:
                raise ValueError(
                    f"No usable observations for validation month "
                    f"{test_month:%Y-%m}."
                )

            model = make_model(model_params, self.seed, self.model_jobs)
            model.fit(train[features], train[self.target])
            forecast = model.predict(test[features])
            errors = test[self.target].to_numpy(dtype=float) - forecast
            fold_sse = float(np.dot(errors, errors))
            squared_error_sum += fold_sse
            observation_count += len(errors)
            fold_rmse = float(np.sqrt(fold_sse / len(errors)))
            monthly_rmse.append(fold_rmse)

            running_rmse = float(
                np.sqrt(squared_error_sum / observation_count)
            )
            trial.report(running_rmse, step=step + 1)
            if trial.should_prune():
                trial.set_user_attr(
                    "validation_months_completed", step + 1
                )
                raise optuna.TrialPruned()

        trial.set_user_attr(
            "validation_months_completed", len(self.validation_months)
        )
        trial.set_user_attr(
            "validation_observations", observation_count
        )
        trial.set_user_attr(
            "monthly_rmse", [round(value, 10) for value in monthly_rmse]
        )
        return float(np.sqrt(squared_error_sum / observation_count))


def random_walk_validation_rmse(data, target, validation_months):
    """Provide the development-sample benchmark without touching the holdout."""
    validation = data[data[DATE_COL].isin(validation_months)]
    actual = validation[target].to_numpy(dtype=float)
    forecast = np.zeros(len(validation), dtype=float)

    if len(actual) == 0:
        return None
    return float(np.sqrt(np.mean(np.square(actual - forecast))))


def iso_date(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def next_month(value):
    return iso_date(pd.Timestamp(value) + pd.offsets.MonthEnd(1))


def study_counts(study):
    states = optuna.trial.TrialState
    return {
        "total": len(study.trials),
        "complete": sum(trial.state == states.COMPLETE for trial in study.trials),
        "pruned": sum(trial.state == states.PRUNED for trial in study.trials),
        "failed": sum(trial.state == states.FAIL for trial in study.trials),
    }


def save_results(
    study,
    output_dir,
    data_path,
    target,
    features,
    design,
    lag_choices,
    n_trials,
    seed,
    tuning_end,
    holdout_months,
    study_name,
    storage,
    benchmark_rmse,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "lightgbm_optuna_best.json"
    trials_path = output_dir / "lightgbm_optuna_trials.csv"

    complete = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete:
        raise RuntimeError(
            "The study finished without a complete trial. Increase the pruning "
            "startup period or inspect failed trials."
        )

    best_params = dict(study.best_trial.params)
    model_params = {
        key: value
        for key, value in best_params.items()
        if key not in {"window_months", "target_lags"}
    }
    holdout = design["holdout_months"]
    development = design["development_months"]
    validation = design["validation_months"]
    ratio = (
        float(study.best_value / benchmark_rmse)
        if benchmark_rmse not in (None, 0.0)
        else None
    )

    payload = {
        "model": "lightgbm",
        "target": target,
        "best_value_rmse": float(study.best_value),
        "best_params": best_params,
        "tuning_end_date": iso_date(development[-1]),
        "recommended_test_start_date": next_month(development[-1]),
        "validation_months": len(validation),
        "reserved_holdout_months": len(holdout),
        "n_trials": len(study.trials),
        "seed": seed,
        "objective": "pooled rolling-validation RMSE",
        "direction": "minimize",
        "best_value": float(study.best_value),
        "best_trial_number": study.best_trial.number,
        "best_model_params": model_params,
        "best_window_months": best_params["window_months"],
        "best_target_lags": best_params["target_lags"],
        "development_random_walk_rmse": benchmark_rmse,
        "best_to_random_walk_rmse_ratio": ratio,
        "data": {
            "path": str(Path(data_path).resolve()),
            "predictor_count_before_added_target_lags": len(features),
            "first_month": iso_date(design["all_months"][0]),
            "last_month": iso_date(design["all_months"][-1]),
        },
        "split": {
            "development_end": iso_date(development[-1]),
            "validation_start": iso_date(validation[0]),
            "validation_end": iso_date(validation[-1]),
            "validation_months": len(validation),
            "recommended_test_start": next_month(development[-1]),
            "reserved_holdout_start": (
                iso_date(holdout[0]) if len(holdout) else None
            ),
            "reserved_holdout_end": (
                iso_date(holdout[-1]) if len(holdout) else None
            ),
            "reserved_holdout_months": len(holdout),
            "boundary_method": (
                f"explicit tuning end {tuning_end}"
                if tuning_end is not None
                else f"reserved final {holdout_months} months"
            ),
        },
        "search": {
            "requested_trials_this_run": n_trials,
            "study_trial_counts": study_counts(study),
            "sampler": "Optuna TPESampler",
            "sampler_seed": seed,
            "pruning_unit": "completed rolling validation month",
            "requested_window_choices": design["requested_windows"],
            "feasible_window_choices": design["feasible_windows"],
            "target_lag_choices": lag_choices,
            "study_name": study_name,
            "storage": storage,
        },
        "protocol": {
            "training_window": "fixed-length months ending before each forecast",
            "validation": "one-month-ahead rolling forecasts in development data",
            "holdout_evaluated_during_tuning": False,
            "selection_rule": "lowest pooled validation RMSE",
            "caution": (
                "The split prevents target leakage from the reserved holdout, "
                "but supplied predictors must also have been constructed "
                "causally for a fully real-time exercise."
            ),
        },
        "software": {
            "optuna": package_version("optuna"),
            "lightgbm": package_version("lightgbm"),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        },
    }

    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    study.trials_dataframe().to_csv(trials_path, index=False)
    return result_path, trials_path, payload


def main(args):
    if args.resume and not args.storage:
        raise ValueError("--resume requires --storage.")
    if args.model_jobs == 0:
        raise ValueError("--model-jobs cannot be zero.")

    if args.smoke_test:
        args.n_trials = 1
        args.validation_months = 2
        args.output_dir = Path(args.output_dir) / "smoke"
        print(
            "Smoke-test mode: 1 trial, 2 validation months, and 20-50 trees."
        )

    data, target, features = load_model_panel(args.data)
    features = exclude_target_lag_features(features, target)
    requested_lags = unique_sorted(args.target_lag_choices)
    design = validation_design(
        data=data,
        tuning_end=args.tuning_end,
        holdout_months=args.holdout_months,
        validation_months=args.validation_months,
        requested_windows=args.window_choices,
    )
    effective_holdout = (
        DEFAULT_HOLDOUT_MONTHS
        if args.tuning_end is None and args.holdout_months is None
        else args.holdout_months
    )

    if design["requested_windows"] != design["feasible_windows"]:
        removed = sorted(
            set(design["requested_windows"])
            - set(design["feasible_windows"])
        )
        print(f"Ignoring infeasible window choices: {removed}")

    print(
        f"Dataset: {Path(args.data).resolve()}\n"
        f"Target: {target}\n"
        f"Development end: {iso_date(design['development_months'][-1])}\n"
        f"Rolling validation: {iso_date(design['validation_months'][0])} "
        f"to {iso_date(design['validation_months'][-1])} "
        f"({len(design['validation_months'])} months)\n"
        f"Reserved holdout: {len(design['holdout_months'])} months\n"
        f"Window choices: {design['feasible_windows']}\n"
        f"Target-lag choices: {requested_lags}"
    )

    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(
        n_startup_trials=args.pruner_startup_trials,
        n_warmup_steps=args.pruner_warmup_months,
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
        base_features=features,
        design=design,
        lag_choices=requested_lags,
        seed=args.seed,
        model_jobs=args.model_jobs,
        smoke_test=args.smoke_test,
    )
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout_seconds,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=not args.no_progress,
    )

    benchmark_rmse = random_walk_validation_rmse(
        data, target, design["validation_months"]
    )
    result_path, trials_path, payload = save_results(
        study=study,
        output_dir=args.output_dir,
        data_path=args.data,
        target=target,
        features=features,
        design=design,
        lag_choices=requested_lags,
        n_trials=args.n_trials,
        seed=args.seed,
        tuning_end=args.tuning_end,
        holdout_months=effective_holdout,
        study_name=args.study_name,
        storage=args.storage,
        benchmark_rmse=benchmark_rmse,
    )

    print(
        f"\nBest validation RMSE: {payload['best_value']:.6f}\n"
        f"Best parameters: {json.dumps(payload['best_params'], sort_keys=True)}\n"
        f"Recommended untouched test start: "
        f"{payload['split']['recommended_test_start']}\n"
        f"Saved best result: {result_path.resolve()}\n"
        f"Saved trial history: {trials_path.resolve()}"
    )


if __name__ == "__main__":
    main(parse_args())

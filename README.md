# Unemployment nowcasting models

The forecasting scripts use monthly rolling windows and keep country and date
identifiers beside every out-of-sample prediction. The target is fixed as the
monthly change in the unemployment rate (`unemployment_change_1m`).

## Reports structure

Generated results are kept under `reports`:

```text
reports/
|-- outputs/                         # Predictions, metrics, models and tuning files
|   `-- tuning/                      # Optuna best settings and trial histories
|-- figures/
|   `-- model_metric_comparison.png  # RMSE, MAE and MAPE comparison
`-- tables/
    `-- model_comparison.xlsx        # One workbook comparing all models
```

Model scripts write their artifacts to `reports/outputs`. Running
`python scripts/analysis.py` reads those predictions and creates the comparison
workbook and figure in `reports/tables` and `reports/figures`, respectively.
The models still forecast monthly unemployment changes. RMSE and MAE evaluate
those changes, while MAPE is calculated from the reconstructed unemployment
rate level so that zero monthly changes do not create undefined percentages.

## Install the model dependencies

```powershell
python -m pip install -r requirements.txt
```

## Tune XGBoost and LightGBM with Optuna

By default, each study reserves the final 24 months as an untouched holdout and
uses the preceding 24 months for rolling validation. Both studies search over
60, 84 and 120-month training windows and zero to three target lags.

```powershell
python scripts/tune_xgboost.py --n-trials 50
python scripts/tune_lightgbm.py --n-trials 50
```

The best settings and full trial histories are saved under
`reports/outputs/tuning`.
Use `--tuning-end YYYY-MM` when an explicit development cutoff is preferred,
or `--timeout-seconds` to impose a computing budget. Long studies can be
resumed by supplying `--storage sqlite:///optuna.db --resume` together with a
stable `--study-name`.

A quick execution check is available for both scripts. Smoke-test scores are
not model results.

```powershell
python scripts/tune_xgboost.py --smoke-test
python scripts/tune_lightgbm.py --smoke-test
```

## Evaluate the protected holdout

The model trainers read the selected window, lags, hyperparameters and holdout
start directly from each Optuna JSON file.

```powershell
python scripts/train_xgboost.py `
  --params-file reports/outputs/tuning/xgboost_optuna_best.json `
  --output-dir reports/outputs

python scripts/train_lightgbm.py `
  --params-file reports/outputs/tuning/lightgbm_optuna_best.json `
  --output-dir reports/outputs
```

With the default 24-month holdout in the current dataset, run the benchmarks
on the same dates and then produce a common-sample comparison:

```powershell
python scripts/train_baseline.py `
  --test-start-date 2024-01 `
  --output-dir reports/outputs

python scripts/analysis.py
```

The protected time split prevents the target holdout from entering Optuna.
Strict real-time claims also require upstream trend, PCA and break adjustments
to be estimated causally or inside each training window.

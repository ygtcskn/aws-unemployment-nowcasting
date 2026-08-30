# AWS Unemployment Nowcasting Project

Can Google search activity help us understand changes in unemployment before the official data is fully available? In this project, I use Google Trends indicators to forecast the monthly change in the unemployment rate.

This is also an **AWS practice project**. I opened an AWS Free Tier account, created an **Amazon S3** bucket and uploaded the raw Google Trends files. I then used an **Amazon EC2** instance to run the full Python pipeline: data preparation, hyperparameter tuning, model estimation and evaluation. The code connects to S3 with `boto3`, while AWS credentials are excluded from Git.

## AWS workflow

```mermaid
flowchart LR
    A[Google Trends CSV files] --> B[Amazon S3<br/>raw/google_trends]
    B --> C[Amazon EC2<br/>Python environment]
    D[OECD unemployment data] --> C
    C --> E[Preprocessing and panel construction]
    E --> F[Optuna tuning]
    F --> G[XGBoost and LightGBM]
    E --> H[AR and Random Walk benchmarks]
    G --> I[Reports, figures and Excel table]
    H --> I
```

The script [`scripts/upload_raw_to_s3.py`](scripts/upload_raw_to_s3.py) creates the S3 bucket in `eu-central-1` and uploads the Google Trends CSV files. The EC2 pipeline then reads these files directly from S3.

I use part of the dataset that I collected for my master's thesis in Economics. The sample includes nine countries: **Australia, Canada, France, Germany, Italy, Japan, South Korea, the United Kingdom and the United States**.

## Data and methodology

For this portfolio project, I apply a shorter version of the preprocessing used in my thesis:

1. Read the monthly Google Trends categories and topics from S3 and create a balanced country panel.
2. Adjust known breaks in the Google Trends methodology by comparing the average level in the 12 months before and after each break.
3. Remove the common search trend within each country using logarithms, an HP filter and the first principal component. This helps control for changes such as the general increase in internet use.
4. Use topics in log levels and categories as 12-month log changes, which also reduces seasonal patterns.
5. Keep the same predictors for every country and add country dummy variables. The affordable-housing topic is excluded because its break adjustment produces an unusable series for Italy.
6. Merge seasonally adjusted OECD unemployment rates and fill isolated one-month gaps using the average of the previous and following months.
7. Construct the dependent variable as the monthly change in unemployment:

   $$
   \Delta u_{i,t}=u_{i,t}-u_{i,t-1}
   $$

The current unemployment rate is removed from the explanatory variables because it would directly reveal the target. All models therefore forecast the same dependent variable.

I compare four models: **Random Walk**, pooled **AR(1)**, **XGBoost** and **LightGBM**. Random Walk and AR(1) are the economic benchmarks. I tune XGBoost and LightGBM with **Optuna**, including their tree parameters, training-window length and number of target lags. The tuning uses rolling validation, while the final 24 months are kept as an untouched test period.

## Results

The table reports results for the final test period using the same 216 country-month observations for every model. RMSE and MAE measure errors in the monthly unemployment change, in percentage points. MAPE is calculated after reconstructing the unemployment-rate level. Lower values are better.

| Rank | Model | RMSE | MAE | MAPE |
|---:|---|---:|---:|---:|
| 1 | Random Walk | **0.1334** | **0.0904** | **2.00%** |
| 2 | LightGBM | 0.1347 | 0.0976 | 2.18% |
| 3 | XGBoost | 0.1377 | 0.1012 | 2.27% |
| 4 | AR(1) | 0.1539 | 0.1071 | 2.38% |

Random Walk performs best in this test period. LightGBM is close, but neither machine-learning model beats the simplest benchmark. This is still an important result: a more complicated model does not necessarily produce a better forecast, so economic benchmarks should always be included.

## Reproduce on EC2

First, the raw Google Trends files can be uploaded from the computer where they are stored:

```bash
python scripts/upload_raw_to_s3.py
```

After cloning the repository on an EC2 instance, configure AWS access and place the OECD unemployment file at `data/raw/unemp/unemp_raw.csv`. Then run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python scripts/prep_data.py
python scripts/tune_xgboost.py --n-trials 50
python scripts/tune_lightgbm.py --n-trials 50
python scripts/train_xgboost.py --params-file reports/outputs/tuning/xgboost_optuna_best.json
python scripts/train_lightgbm.py --params-file reports/outputs/tuning/lightgbm_optuna_best.json
python scripts/train_baseline.py --test-start-date 2024-01
python scripts/analysis.py
```

The pipeline saves predictions and fitted models in `reports/outputs`, figures in [`reports/figures`](reports/figures), and the comparison workbook in [`reports/tables/model_comparison.xlsx`](reports/tables/model_comparison.xlsx).

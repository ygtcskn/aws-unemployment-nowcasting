"""Prepare the monthly country panel used for unemployment nowcasting.

The script cleans Google Trends data, builds a balanced panel, and adds a
seasonally adjusted OECD unemployment target. Run it from the project root.
"""

import argparse
import re
from collections import Counter
from io import StringIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.filters.hp_filter import hpfilter
from tqdm import tqdm


# --- Research sample and file paths ---

BUCKET_NAME = "aws-unemp-nowcast-ygtcskn"
REGION = "eu-central-1"

RAW_PREFIX = "raw/google_trends/"

FINAL_DIR = Path("data/final")
UNEMPLOYMENT_FILE = Path("data/raw/unemp/unemp_raw.csv")
UNEMPLOYMENT_FINAL_FILE = FINAL_DIR / "gt_unemp_monthly_final.csv"
UNEMPLOYMENT_TARGET_MODES = ("change_1m", "raw")
DEFAULT_UNEMPLOYMENT_TARGET = "change_1m"

# These exclusions define the estimation sample. The affordable-housing topic
# is removed because its break adjustment collapses the Italian series to zero.
DROP_COUNTRIES = {"CN", "SA", "TR", "ZA", "MX", "RU", "ID", "IN", "AR", "BR"}
DROP_SERIES_IDS = {"10052"}

# Australia is the reference country in pooled panel regressions.
DUMMY_REFERENCE_COUNTRY = "AU"

UNEMPLOYMENT_COUNTRY_MAP = {
    "AUS": "AU",
    "CAN": "CA",
    "DEU": "DE",
    "FRA": "FR",
    "GBR": "GB",
    "ITA": "IT",
    "JPN": "JP",
    "KOR": "KR",
    "USA": "US",
}

# These OECD dimensions select comparable headline rates: total population
# aged 15+, percentage of the labour force, seasonally and calendar adjusted.
UNEMPLOYMENT_DIMENSIONS = {
    "MEASURE": "UNE_LF_M",
    "UNIT_MEASURE": "PT_LF_SUB",
    "TRANSFORMATION": "_Z",
    "ADJUSTMENT": "Y",
    "SEX": "_T",
    "AGE": "Y_GE15",
    "ACTIVITY": "_Z",
    "FREQ": "M",
}

CATEGORY_BREAK_YEARS = (2011, 2016, 2022)
TOPIC_BREAK_YEARS = (2011, 2016, 2017, 2022)
BREAK_WINDOW_MONTHS = 12
MIN_BREAK_LEVEL = 1e-6
MIN_BREAK_CHANGE = 0.01

HP_LAMBDA_MONTHLY = 129600
YOY_LAG_MONTHLY = 12
LOG_FLOOR = 1e-6

MONTHLY_FILENAME_RE = re.compile(
    r"^(?P<series_id>\d+)_(?P<country>[A-Z]{2})_m_[^.]+\.csv$",
    flags=re.IGNORECASE,
)

# Readable labels for the Google Trends categories and topics used in the thesis.
CAT_MAP = {
    "3": "arts_entertainment", "5": "computers_electronics", "7": "finance",
    "8": "games", "11": "home_garden", "12": "business_industrial",
    "13": "internet_telecom", "14": "people_society", "16": "news",
    "18": "shopping", "19": "law_government", "20": "sports",
    "22": "books_literature", "23": "performing_arts", "24": "visual_art_design",
    "25": "advertising_marketing", "28": "office_services", "29": "real_estate",
    "30": "computer_hardware", "31": "programming", "32": "software",
    "33": "offbeat", "34": "movies", "35": "music_audio", "36": "tv_video",
    "37": "banking", "38": "insurance", "39": "card_games",
    "41": "computer_video_games", "43": "online_goodies", "44": "beauty_fitness",
    "45": "health", "46": "agriculture_forestry", "47": "autos_vehicles",
    "48": "construction_maintenance", "49": "manufacturing",
    "50": "transportation_logistics", "53": "web_hosting_domain_registration",
    "54": "social_issues_advocacy", "55": "dating_personals",
    "56": "ethnic_identity_groups", "57": "charity_philanthropy",
    "59": "religion_belief", "60": "jobs", "64": "antiques_collectibles",
    "65": "hobbies_leisure", "66": "pets_animals", "67": "travel",
    "68": "apparel", "69": "consumer_resources", "70": "gifts_special_event_items",
    "71": "food_drink", "73": "mass_merchants_department_stores", "74": "education",
    "75": "legal", "76": "government", "77": "enterprise_technology",
    "78": "consumer_electronics", "82": "environmental_issues",
    "83": "marketing_services", "84": "search_engine_optimization_marketing",
    "89": "vehicle_parts_accessories", "93": "skin_nail_care", "94": "fitness",
    "95": "office_supplies", "96": "real_estate_agencies",
    "97": "consumer_advocacy_protection", "99": "gifts",
    "121": "grocery_food_retailers", "123": "tobacco_products",
    "144": "unwanted_body_facial_hair_removal", "145": "spas_beauty_services",
    "158": "home_improvement", "168": "emergency_services",
    "170": "vehicle_licensing_registration", "179": "hotels_accommodations",
    "205": "car_rental_taxi_services", "206": "cruises_charters",
    "208": "tourist_destinations", "233": "energy_utilities", "248": "pharmacy",
    "249": "health_insurance", "250": "hospitals_treatment_centers",
    "255": "pharmaceuticals_biotech", "256": "medical_facilities_services",
    "270": "home_furnishings", "271": "home_appliances", "276": "restaurants",
    "277": "alcoholic_beverages", "278": "accounting_auditing",
    "279": "credit_lending", "287": "industrial_materials_equipment",
    "288": "chemicals_industry", "289": "freight_trucking", "290": "packaging",
    "291": "moving_relocation", "293": "weddings", "314": "computer_security",
    "329": "business_services", "334": "corporate_events",
    "341": "customer_relationship_management_crm",
    "342": "enterprise_resource_planning_erp", "343": "data_management",
    "354": "import_export", "355": "book_retailers", "365": "coupons_discount_offers",
    "378": "apartments_residential_rentals", "380": "veterinarians", "396": "politics",
    "408": "newspapers", "423": "bankruptcy", "425": "property_management",
    "437": "mental_health", "465": "home_insurance", "466": "home_financing",
    "468": "auto_financing", "473": "vehicle_shopping", "477": "architecture",
    "508": "social_services", "566": "textiles_nonwovens", "569": "events_listings",
    "606": "metals_mining", "610": "trucks_suvs", "612": "entertainment_industry",
    "621": "food_production", "650": "building_materials_supplies",
    "651": "civil_engineering", "652": "construction_consulting_contracting",
    "657": "renewable_alt_energy", "658": "electricity", "659": "oil_gas",
    "660": "waste_management", "662": "aviation", "664": "distribution_logistics",
    "665": "maritime_transport", "666": "rail_transport", "667": "urban_transport",
    "670": "agrochemicals", "672": "coatings_adhesives", "673": "dyes_pigments",
    "696": "luxury_goods", "697": "footwear", "718": "outsourcing",
    "728": "computer_servers", "730": "development_tools", "747": "aquaculture",
    "748": "agricultural_equipment", "750": "forestry", "784": "business_news",
    "794": "gps_navigation", "802": "developer_jobs", "813": "college_financing",
    "815": "vehicle_brands", "832": "flooring", "841": "retail_trade",
    "882": "animal_products_services", "894": "acting_theater", "918": "fast_food",
    "952": "swimming_pools_spas", "960": "job_listings", "961": "resumes_portfolios",
    "969": "legal_services", "1003": "luggage_travel_accessories",
    "1010": "travel_agencies_services", "1080": "real_estate_listings",
    "1081": "timeshares_vacation_properties", "1140": "boats_watercraft",
    "1143": "entertainment_media", "1159": "business_operations",
    "1160": "commercial_lending", "1162": "consulting", "1164": "economy_news",
    "1176": "printing_publishing", "1188": "car_electronics",
    "1199": "professional_trade_associations", "1209": "world_news",
    "1214": "commercial_vehicles", "1268": "fuel_economy_gas_prices",
    "1269": "vehicle_fuels_lubricants", "1300": "cad_cam", "1306": "parking",
    "1339": "carpooling_ridesharing", "1349": "water_supply_treatment",
    "10001": "t_inflation", "10002": "t_cost_of_living",
    "10003": "t_consumer_price_index", "10004": "t_price",
    "10005": "t_purchasing_power", "10006": "t_cheap",
    "10007": "t_interest_rate", "10008": "t_central_bank",
    "10009": "t_monetary_policy", "10010": "t_exchange_rate",
    "10011": "t_currency", "10012": "t_foreign_exchange_market",
    "10013": "t_rent", "10014": "t_real_estate", "10015": "t_housing",
    "10016": "t_electricity", "10017": "t_natural_gas", "10018": "t_gasoline",
    "10019": "t_petroleum", "10020": "t_invoice", "10021": "t_public_transport",
    "10022": "t_auto_insurance", "10023": "t_grocery_store", "10024": "t_bread",
    "10025": "t_milk", "10026": "t_eggs", "10027": "t_meat", "10028": "t_coffee",
    "10029": "t_restaurant", "10030": "t_wage", "10031": "t_salary",
    "10032": "t_minimum_wage", "10033": "t_strike_action", "10034": "t_unemployment",
    "10036": "t_savings", "10038": "t_debt", "10039": "t_birthday",
    "10041": "t_unemployment_benefits", "10042": "t_recruitment",
    "10043": "t_investment", "10044": "t_lawyer", "10045": "t_jobs",
    "10046": "t_economic_crisis", "10047": "t_financial_crisis",
    "10048": "t_house_price_index", "10049": "t_crisis", "10050": "t_interest",
    "10051": "t_student_loan", "10052": "t_affordable_housing",
    "10053": "t_recession", "10054": "t_bankruptcy", "10055": "t_export",
    "10056": "t_baggage", "10057": "t_liquidation",
}


# --- Google Trends input ---

def feature_name(series_id):
    if series_id in CAT_MAP:
        return CAT_MAP[series_id]
    prefix = "t" if int(series_id) >= 10000 else "cat"
    return f"{prefix}_{series_id}"


def parse_monthly_key(key):
    filename = key.rsplit("/", 1)[-1]
    match = MONTHLY_FILENAME_RE.fullmatch(filename)
    if not match:
        return None
    return {
        "key": key,
        "series_id": match.group("series_id"),
        "country": match.group("country").upper(),
    }


def list_monthly_objects(s3_client):
    paginator = s3_client.get_paginator("list_objects_v2")
    objects = []

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=RAW_PREFIX):
        for item in page.get("Contents", []):
            parsed = parse_monthly_key(item["Key"])
            if parsed is not None:
                objects.append(parsed)

    objects.sort(key=lambda item: item["key"])
    if not objects:
        raise FileNotFoundError(
            f"No monthly files matching '<id>_<CC>_m_<period>.csv' were found "
            f"under s3://{BUCKET_NAME}/{RAW_PREFIX}"
        )

    return objects


def read_monthly_csv_from_s3(s3_client, key):
    """Read one monthly series and standardize its date and value columns."""
    response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
    raw_bytes = response["Body"].read()

    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")

    try:
        df = pd.read_csv(
            StringIO(text),
            skiprows=2,
            header=None,
            usecols=[0, 1],
            names=["date", "value"],
        )
    except Exception as exc:
        raise ValueError(f"Could not parse s3://{BUCKET_NAME}/{key}: {exc}") from exc

    df["date"] = pd.to_datetime(
        df["date"].astype(str).str.strip(),
        format="%Y-%m",
        errors="coerce",
    )

    values = df["value"].astype(str).str.strip().replace({"<1": "0.5"})
    df["value"] = pd.to_numeric(values, errors="coerce")
    df = df.dropna(subset=["date", "value"])

    if df.empty:
        raise ValueError(f"No usable monthly observations in s3://{BUCKET_NAME}/{key}")

    if df["date"].duplicated().any():
        duplicated = df.loc[
            df["date"].duplicated(keep=False), "date"
        ].dt.strftime("%Y-%m")
        examples = sorted(duplicated.unique())[:5]
        raise ValueError(f"Duplicate monthly dates in {key}: {examples}")

    return df.set_index("date")["value"].sort_index()


def load_monthly_matrix(s3_client):
    """Build the wide country-feature matrix for the chosen research sample."""
    objects = list_monthly_objects(s3_client)
    dropped_file_counts = Counter()
    dropped_series_file_counts = Counter()
    retained = []

    for item in objects:
        if item["series_id"] in DROP_SERIES_IDS:
            dropped_series_file_counts[item["series_id"]] += 1
        elif item["country"] in DROP_COUNTRIES:
            dropped_file_counts[item["country"]] += 1
        else:
            retained.append(item)

    if not retained:
        raise ValueError("No Google Trends files remain after the sample exclusions.")

    column_names = [
        f"{item['country']}_{feature_name(item['series_id'])}"
        for item in retained
    ]
    duplicates = sorted(
        name for name, count in Counter(column_names).items() if count > 1
    )
    if duplicates:
        sample = ", ".join(duplicates[:10])
        raise ValueError(
            "More than one monthly export maps to the same country-feature. "
            f"Keep one long-run export per series. Examples: {sample}"
        )

    series = {}
    for item in tqdm(retained, desc="Reading monthly GT files", unit="file"):
        column = f"{item['country']}_{feature_name(item['series_id'])}"
        series[column] = read_monthly_csv_from_s3(s3_client, item["key"])

    matrix = pd.DataFrame(series).sort_index()
    matrix.index.name = "date"

    return matrix, objects, dropped_file_counts, dropped_series_file_counts


# --- Structural-break adjustment ---

def is_topic(column):
    parts = column.split("_", 1)
    return len(parts) == 2 and parts[1].startswith("t_")


def window_mean(series, break_date, side):
    if side == "before":
        window = series[
            (series.index < break_date)
            & (series.index >= break_date - pd.DateOffset(months=BREAK_WINDOW_MONTHS))
        ]
    else:
        window = series[
            (series.index >= break_date)
            & (series.index < break_date + pd.DateOffset(months=BREAK_WINDOW_MONTHS))
        ]
    return window.mean() if not window.empty else np.nan


def adjust_series_breaks(series, break_years):
    """Rescale known platform breaks without changing within-period variation.

    Google methodology changes can look like economic shocks. Comparing the
    twelve months on each side keeps those artificial level shifts out of the
    forecasting signal.
    """
    adjusted = series.dropna().copy().astype(float)

    for break_year in sorted(break_years):
        break_date = pd.Timestamp(f"{break_year}-01-01")
        if (
            adjusted.empty
            or break_date < adjusted.index.min()
            or break_date > adjusted.index.max()
        ):
            continue

        mean_before = window_mean(adjusted, break_date, side="before")
        mean_after = window_mean(adjusted, break_date, side="after")

        if not np.isfinite(mean_before) or not np.isfinite(mean_after):
            continue
        if mean_after < MIN_BREAK_LEVEL:
            continue

        ratio = mean_before / mean_after
        if abs(ratio - 1.0) < MIN_BREAK_CHANGE:
            continue

        adjusted.loc[adjusted.index >= break_date] *= ratio

    return adjusted


def adjust_breaks(matrix):
    """Apply the relevant break calendar to every search series."""
    adjusted = {}
    for column in tqdm(matrix.columns, desc="Adjusting monthly breaks", unit="series"):
        years = TOPIC_BREAK_YEARS if is_topic(column) else CATEGORY_BREAK_YEARS
        adjusted[column] = adjust_series_breaks(matrix[column], years)
    result = pd.DataFrame(adjusted, index=matrix.index).sort_index()
    result.index.name = "date"
    return result


# --- Country-level common-trend removal ---

def shift_to_positive(matrix):
    """Shift a series only when needed so logarithms are well defined."""
    shifted = matrix.copy().astype(float)
    for column in shifted.columns:
        minimum = shifted[column].min(skipna=True)
        if not np.isfinite(minimum):
            raise ValueError(f"Series has no finite values: {column}")
        if minimum < 1.0:
            shifted[column] = shifted[column] + (1.0 - minimum)
    return shifted


def validate_complete_country_matrix(matrix, country):
    missing = matrix.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        details = ", ".join(f"{col}={count}" for col, count in missing.head(10).items())
        raise ValueError(
            f"{country} has missing monthly observations before detrending. "
            f"HP/PCA requires a balanced time series. Missing counts: {details}"
        )

    expected_index = pd.date_range(matrix.index.min(), matrix.index.max(), freq="MS")
    if not matrix.index.equals(expected_index):
        missing_dates = (
            expected_index.difference(matrix.index).strftime("%Y-%m").tolist()
        )
        raise ValueError(
            f"{country} does not have a complete monthly calendar. "
            f"Missing dates include: {missing_dates[:10]}"
        )


def detrend_country(country_matrix, country):
    """Remove the broad search trend shared by features within one country.

    The HP filter extracts slow-moving trends, while the first principal
    component summarizes common changes such as wider internet or Google use.
    Removing it leaves more of the topic-specific economic signal.
    """
    validate_complete_country_matrix(country_matrix, country)
    country_matrix = shift_to_positive(country_matrix)

    # Constant series stay in the panel but do not enter the HP/PCA estimation.
    nonconstant = country_matrix.nunique(dropna=True) > 1
    variable_matrix = country_matrix.loc[:, nonconstant].copy()
    constant_matrix = country_matrix.loc[:, ~nonconstant].copy()
    if variable_matrix.empty:
        return country_matrix

    logged = np.log(variable_matrix)
    trends = {}
    for column in logged.columns:
        _, trend = hpfilter(logged[column], lamb=HP_LAMBDA_MONTHLY)
        trends[column] = trend

    trend_matrix = pd.DataFrame(trends, index=logged.index)
    scaled_trends = StandardScaler().fit_transform(trend_matrix)
    pc1 = pd.Series(
        PCA(n_components=1).fit_transform(scaled_trends)[:, 0],
        index=trend_matrix.index,
        dtype=float,
    )

    average_trend_std = trend_matrix.mean(axis=1).std()
    if not np.isfinite(pc1.std()) or pc1.std() < 1e-6:
        pc1_rescaled = pc1 * 0.0
    else:
        pc1_rescaled = (pc1 - pc1.mean()) / pc1.std() * average_trend_std

    filtered_log = logged.sub(pc1_rescaled, axis=0)
    detrended = np.exp(filtered_log)

    scale_factors = variable_matrix.mean() / detrended.mean().replace(0, np.nan)
    detrended = detrended * scale_factors.fillna(1.0)
    return pd.concat([detrended, constant_matrix], axis=1)[country_matrix.columns]


def detrend_all_countries(matrix):
    """Detrend each country separately so national search patterns are respected."""
    countries = sorted({column.split("_", 1)[0] for column in matrix.columns})
    outputs = []

    for country in tqdm(countries, desc="Detrending monthly GT", unit="country"):
        columns = [
            column
            for column in matrix.columns
            if column.startswith(f"{country}_")
        ]
        country_output = detrend_country(matrix[columns], country)
        outputs.append(country_output)

    result = pd.concat(outputs, axis=1).sort_index()
    result.index.name = "date"
    return result


# --- Econometric transformations ---


def transform_monthly(matrix):
    """Transform searches into the variables used in the forecasting models.

    Topics enter in log levels. Categories enter as twelve-month log changes,
    which removes recurring seasonality and gives an approximate growth rate.
    """
    logged = np.log(matrix.clip(lower=LOG_FLOOR))
    transformed = pd.DataFrame(index=matrix.index, columns=matrix.columns, dtype=float)

    for column in matrix.columns:
        if is_topic(column):
            transformed[column] = logged[column]
        else:
            transformed[column] = logged[column] - logged[column].shift(YOY_LAG_MONTHLY)

    transformed.index.name = "date"
    return transformed


# --- Balanced country panel ---


def validate_feature_coverage(matrix):
    """Require the same predictors for every country in the pooled sample."""
    coverage = {}
    countries = sorted({column.split("_", 1)[0] for column in matrix.columns})
    for country in countries:
        prefix = f"{country}_"
        coverage[country] = {
            column[len(prefix):]
            for column in matrix.columns
            if column.startswith(prefix)
        }

    if not coverage:
        raise ValueError("No country features are available for panel construction.")

    union = set().union(*coverage.values())
    intersection = set.intersection(*coverage.values())

    if union != intersection:
        lines = []
        for country, features in coverage.items():
            missing = sorted(union - features)
            extra = sorted(features - intersection)
            if missing or extra:
                lines.append(
                    f"{country}: {len(features)} features, "
                    f"missing {missing[:5]}, non-common {extra[:5]}"
                )
        raise ValueError(
            "Retained countries do not have identical feature coverage.\n"
            + "\n".join(lines)
        )

    return sorted(intersection)


def build_monthly_panel(matrix):
    """Stack countries into a monthly panel and add country fixed effects."""
    features = validate_feature_coverage(matrix)
    countries = sorted({column.split("_", 1)[0] for column in matrix.columns})
    country_blocks = []

    for country in countries:
        columns = [f"{country}_{feature}" for feature in features]
        block = matrix[columns].copy()
        block.columns = features
        block.insert(0, "country", country)
        block.insert(0, "date", block.index + pd.offsets.MonthEnd(0))
        block = block.reset_index(drop=True)
        country_blocks.append(block)

    panel = pd.concat(country_blocks, ignore_index=True)

    before_drop = len(panel)
    panel = panel.dropna(subset=features).copy()
    incomplete_rows_dropped = before_drop - len(panel)

    if panel.empty:
        raise ValueError(
            "No complete country-month rows remain after the transformations."
        )

    if DUMMY_REFERENCE_COUNTRY in countries:
        dummy_order = [DUMMY_REFERENCE_COUNTRY] + [
            country for country in countries if country != DUMMY_REFERENCE_COUNTRY
        ]
    else:
        dummy_order = countries

    # Dropping Australia avoids the dummy-variable trap in linear models.
    country_category = pd.Categorical(
        panel["country"], categories=dummy_order, ordered=True
    )
    dummies = pd.get_dummies(
        country_category,
        prefix="C",
        drop_first=True,
        dtype=int,
    )
    dummies.index = panel.index
    panel = pd.concat([panel, dummies], axis=1)
    panel = panel.sort_values(["date", "country"]).reset_index(drop=True)

    return panel, features, incomplete_rows_dropped, dummy_order[0]


def validate_final_panel(panel, feature_columns, reference_country):
    present_countries = sorted(panel["country"].unique())
    forbidden = sorted(set(present_countries) & DROP_COUNTRIES)
    if forbidden:
        raise AssertionError(f"Excluded countries are still present: {forbidden}")

    duplicates = panel.duplicated(["date", "country"], keep=False)
    if duplicates.any():
        examples = panel.loc[duplicates, ["date", "country"]].head(5)
        raise AssertionError(
            f"Duplicate country-month keys found: {examples.to_dict('records')}"
        )

    if panel[feature_columns].isna().any().any():
        raise AssertionError("Final model features contain missing values.")

    values = panel[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("Final model features contain non-finite values.")

    dummy_columns = [column for column in panel.columns if column.startswith("C_")]
    expected_dummies = max(len(present_countries) - 1, 0)
    if len(dummy_columns) != expected_dummies:
        raise AssertionError(
            f"Expected {expected_dummies} dummies, found {len(dummy_columns)}."
        )

    return {
        "countries": present_countries,
        "country_count": len(present_countries),
        "feature_count": len(feature_columns),
        "dummy_count": len(dummy_columns),
        "reference_country": reference_country,
        "rows": len(panel),
        "start": panel["date"].min(),
        "end": panel["date"].max(),
    }


# --- OECD unemployment target ---

def load_unemployment_rates(path=UNEMPLOYMENT_FILE):
    """Load one comparable, seasonally adjusted unemployment series per country."""
    if not path.exists():
        raise FileNotFoundError(f"Unemployment source file not found: {path.resolve()}")

    source = pd.read_csv(path, low_memory=False)
    required_columns = {
        "REF_AREA",
        "TIME_PERIOD",
        "OBS_VALUE",
        *UNEMPLOYMENT_DIMENSIONS,
    }
    missing_columns = sorted(required_columns - set(source.columns))
    if missing_columns:
        raise ValueError(
            f"Unemployment source is missing required columns: {missing_columns}"
        )

    dimension_errors = {}
    for column, expected in UNEMPLOYMENT_DIMENSIONS.items():
        observed = sorted(source[column].dropna().astype(str).unique())
        if observed != [expected]:
            dimension_errors[column] = observed
    if dimension_errors:
        raise ValueError(
            "Unemployment source does not contain only the configured OECD "
            f"headline series: {dimension_errors}"
        )

    unemployment = source[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
    unemployment["country"] = unemployment["REF_AREA"].map(
        UNEMPLOYMENT_COUNTRY_MAP
    )
    unknown_areas = sorted(
        unemployment.loc[unemployment["country"].isna(), "REF_AREA"]
        .dropna()
        .unique()
    )
    if unknown_areas:
        raise ValueError(f"No alpha-2 country mapping for OECD areas: {unknown_areas}")

    unemployment["date"] = pd.to_datetime(
        unemployment["TIME_PERIOD"],
        format="%Y-%m",
        errors="coerce",
    ) + pd.offsets.MonthEnd(0)
    unemployment["unemployment_rate"] = pd.to_numeric(
        unemployment["OBS_VALUE"],
        errors="coerce",
    )

    invalid = unemployment[
        unemployment[["country", "date", "unemployment_rate"]].isna().any(axis=1)
    ]
    if not invalid.empty:
        raise ValueError(
            f"Unemployment source has {len(invalid)} rows with invalid keys or values."
        )

    duplicates = unemployment.duplicated(["date", "country"], keep=False)
    if duplicates.any():
        examples = unemployment.loc[duplicates, ["date", "country"]].head(10)
        raise ValueError(
            "Duplicate unemployment country-month observations found: "
            f"{examples.to_dict('records')}"
        )

    return (
        unemployment[["date", "country", "unemployment_rate"]]
        .sort_values(["date", "country"])
        .reset_index(drop=True)
    )


def combine_with_unemployment(panel, unemployment, target_mode):
    """Merge unemployment and construct the selected forecasting target."""
    if target_mode not in UNEMPLOYMENT_TARGET_MODES:
        raise ValueError(
            f"Unknown unemployment target mode {target_mode!r}; "
            f"choose from {sorted(UNEMPLOYMENT_TARGET_MODES)}."
        )

    panel_countries = set(panel["country"].unique())
    unemployment_countries = set(unemployment["country"].unique())
    missing_countries = sorted(panel_countries - unemployment_countries)
    if missing_countries:
        raise ValueError(
            f"Unemployment source is missing retained countries: {missing_countries}"
        )

    relevant_unemployment = unemployment[
        unemployment["country"].isin(panel_countries)
    ]
    combined = panel.merge(
        relevant_unemployment,
        on=["date", "country"],
        how="left",
        validate="one_to_one",
    ).sort_values(["country", "date"]).copy()

    grouped = combined.groupby("country", sort=False)
    previous_rate = grouped["unemployment_rate"].shift(1)
    following_rate = grouped["unemployment_rate"].shift(-1)
    previous_date = grouped["date"].shift(1)
    following_date = grouped["date"].shift(-1)
    current_month = combined["date"].dt.year * 12 + combined["date"].dt.month
    previous_month = previous_date.dt.year * 12 + previous_date.dt.month
    following_month = following_date.dt.year * 12 + following_date.dt.month

    # An isolated missing month is filled from its two neighbours. This keeps
    # the monthly change interpretable without extrapolating longer data gaps.
    interpolated_values = (previous_rate + following_rate) / 2.0
    interpolated_mask = (
        combined["unemployment_rate"].isna()
        & previous_rate.notna()
        & following_rate.notna()
        & current_month.sub(previous_month).eq(1)
        & following_month.sub(current_month).eq(1)
    )
    interpolated_rate_keys = combined.loc[
        interpolated_mask, ["date", "country"]
    ].copy()
    interpolated_rate_keys["imputed_value"] = interpolated_values.loc[
        interpolated_mask
    ].to_numpy()
    combined.loc[interpolated_mask, "unemployment_rate"] = interpolated_values.loc[
        interpolated_mask
    ]

    remaining_rate_gaps = combined.loc[
        combined["unemployment_rate"].isna(), ["date", "country"]
    ].copy()

    if target_mode == "change_1m":
        # In changes, a random walk is simply a zero-change benchmark. This is
        # the preferred target for LASSO and tree-based forecasting models.
        grouped = combined.groupby("country", sort=False)
        previous_rate = grouped["unemployment_rate"].shift(1)
        adjacent_month = current_month.sub(previous_month).eq(1)

        target_column = "unemployment_change_1m"
        combined[target_column] = (
            combined["unemployment_rate"] - previous_rate
        ).where(adjacent_month)
        combined = combined.drop(columns=["unemployment_rate"])
    else:
        target_column = "unemployment_rate"

    before_drop = len(combined)
    combined = combined.dropna(subset=[target_column]).copy()
    target_rows_dropped = before_drop - len(combined)
    combined = combined.sort_values(["date", "country"]).reset_index(drop=True)

    if combined.empty:
        raise ValueError(f"No rows remain for unemployment target {target_mode!r}.")

    return (
        combined,
        target_column,
        target_rows_dropped,
        interpolated_rate_keys,
        remaining_rate_gaps,
    )


def validate_combined_panel(
    panel,
    feature_columns,
    reference_country,
    target_column,
):
    summary = validate_final_panel(panel, feature_columns, reference_country)

    if panel.isna().any().any():
        raise AssertionError("Combined panel contains missing values.")

    target = panel[target_column].to_numpy(dtype=float)
    if not np.isfinite(target).all():
        raise AssertionError(f"Target {target_column} contains non-finite values.")

    summary["target_column"] = target_column
    summary["target_min"] = float(target.min())
    summary["target_max"] = float(target.max())
    return summary


# --- Command-line entry point ---

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unemployment-target",
        choices=UNEMPLOYMENT_TARGET_MODES,
        default=DEFAULT_UNEMPLOYMENT_TARGET,
        help=(
            "Target to append: 'change_1m' (recommended percentage-point "
            "change) or 'raw' (seasonally adjusted rate). Both modes write "
            "the same configured final filename."
        ),
    )
    return parser.parse_args()


def main(unemployment_target=DEFAULT_UNEMPLOYMENT_TARGET):
    """Run the complete preparation pipeline and save the modeling panel."""
    s3_client = boto3.client("s3", region_name=REGION)

    (
        monthly_raw,
        objects,
        dropped_file_counts,
        dropped_series_file_counts,
    ) = load_monthly_matrix(s3_client)
    monthly_break_adjusted = adjust_breaks(monthly_raw)
    monthly_detrended = detrend_all_countries(monthly_break_adjusted)
    monthly_transformed = transform_monthly(monthly_detrended)

    panel, features, incomplete_rows_dropped, reference = build_monthly_panel(
        monthly_transformed
    )
    gt_summary = validate_final_panel(panel, features, reference)

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    unemployment = load_unemployment_rates()
    (
        combined,
        target_column,
        target_rows_dropped,
        interpolated_rate_keys,
        remaining_rate_gaps,
    ) = combine_with_unemployment(panel, unemployment, unemployment_target)
    summary = validate_combined_panel(
        combined,
        features,
        reference,
        target_column,
    )
    combined.to_csv(UNEMPLOYMENT_FINAL_FILE, index=False)

    print("\n" + "=" * 72)
    print("UNEMPLOYMENT NOWCASTING PANEL COMPLETE")
    print(f"S3 source: s3://{BUCKET_NAME}/{RAW_PREFIX}")
    print(f"Monthly files discovered: {len(objects):,}")
    print(f"Files excluded by country: {sum(dropped_file_counts.values()):,}")
    print(f"Excluded file counts: {dict(sorted(dropped_file_counts.items()))}")
    print(f"Files excluded by series: {sum(dropped_series_file_counts.values()):,}")
    print(
        "Excluded series file counts: "
        f"{dict(sorted(dropped_series_file_counts.items()))}"
    )
    print(
        f"Retained countries ({gt_summary['country_count']}): "
        f"{gt_summary['countries']}"
    )
    print(f"Confirmed absent: {sorted(DROP_COUNTRIES)}")
    print(f"Google Trends features: {gt_summary['feature_count']:,}")
    print(
        f"Country dummies: {gt_summary['dummy_count']} "
        f"(reference: {gt_summary['reference_country']})"
    )
    print(
        "Rows removed for incomplete 12-month transforms: "
        f"{incomplete_rows_dropped:,}"
    )
    print(f"Google Trends rows: {gt_summary['rows']:,}")
    print(
        f"Google Trends date range: {gt_summary['start'].date()} "
        f"to {gt_summary['end'].date()}"
    )

    print("-" * 72)
    print(f"Unemployment target mode: {unemployment_target}")
    print(f"Target column: {summary['target_column']}")
    print(f"Interpolated one-month rate gaps: {len(interpolated_rate_keys):,}")
    if not interpolated_rate_keys.empty:
        examples = [
            f"{row.country}:{row.date.strftime('%Y-%m')}={row.imputed_value:.3f}"
            for row in interpolated_rate_keys.head(10).itertuples(index=False)
        ]
        print(f"Interpolation examples: {examples}")
    print(f"Unfilled source rate gaps: {len(remaining_rate_gaps):,}")
    print(f"Rows unavailable for target: {target_rows_dropped:,}")
    print(f"Combined rows: {summary['rows']:,}")
    print(f"Target range: {summary['target_min']:.3f} to {summary['target_max']:.3f}")
    print(f"Saved to: {UNEMPLOYMENT_FINAL_FILE.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    args = parse_args()
    main(args.unemployment_target)

"""
src/data_prep.py
================
Data preparation module for Retail Demand Forecasting.

Handles:
- Synthetic data generation (mirrors Rossmann schema)
- Missing value imputation
- Datetime parsing & feature engineering
- Weekly/daily resampling
- Train/test split (time-based)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SYNTHETIC DATA GENERATOR  (mirrors Rossmann schema)
# ══════════════════════════════════════════════════════════════════════════════
def generate_rossmann_sample(
    n_stores: int = 5,
    start: str = "2020-01-01",
    end: str = "2023-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic dataset matching the Rossmann Store Sales schema.
    Incorporates realistic trend, weekly seasonality, holiday effects, and noise.

    Parameters
    ----------
    n_stores : int   Number of stores to simulate
    start    : str   Start date (YYYY-MM-DD)
    end      : str   End date   (YYYY-MM-DD)
    seed     : int   Random seed for reproducibility

    Returns
    -------
    pd.DataFrame  Raw sales DataFrame
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start, end=end, freq="D")
    rows = []

    store_meta = {
        i: {
            "StoreType": np.random.choice(["a", "b", "c", "d"]),
            "Assortment": np.random.choice(["a", "b", "c"]),
            "CompetitionDistance": np.random.randint(100, 10_000),
        }
        for i in range(1, n_stores + 1)
    }

    holiday_dates = {
        pd.Timestamp("2020-12-25"), pd.Timestamp("2021-12-25"),
        pd.Timestamp("2022-12-25"), pd.Timestamp("2023-12-25"),
        pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01"),
        pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01"),
        pd.Timestamp("2020-04-10"), pd.Timestamp("2021-04-02"),
        pd.Timestamp("2022-04-15"), pd.Timestamp("2023-04-07"),
    }
    school_holidays = set(
        pd.date_range("2020-07-01", "2020-08-31").union(
        pd.date_range("2021-07-01", "2021-08-31")).union(
        pd.date_range("2022-07-01", "2022-08-31")).union(
        pd.date_range("2023-07-01", "2023-08-31"))
    )

    for store_id, meta in store_meta.items():
        base_sales = np.random.randint(4_000, 12_000)

        for i, date in enumerate(dates):
            # Closed on Sundays & state holidays
            if date.dayofweek == 6 or date in holiday_dates:
                rows.append({
                    "Store": store_id, "Date": date, "Sales": 0,
                    "Customers": 0, "Open": 0, "Promo": 0,
                    "StateHoliday": "a" if date in holiday_dates else "0",
                    "SchoolHoliday": int(date in school_holidays),
                    **meta,
                })
                continue

            # --- Seasonality components ---
            trend          = base_sales + i * 0.3                          # slow upward trend
            weekly_season  = base_sales * 0.12 * np.sin(2*np.pi * date.dayofweek / 7)
            annual_season  = base_sales * 0.20 * np.sin(2*np.pi * date.dayofyear / 365 - np.pi/2)
            promo          = np.random.choice([0, 1], p=[0.55, 0.45])
            promo_lift     = base_sales * 0.18 * promo
            holiday_lift   = base_sales * 0.08 * int(date in school_holidays)
            noise          = np.random.normal(0, base_sales * 0.06)

            sales = max(0, int(trend + weekly_season + annual_season + promo_lift + holiday_lift + noise))
            customers = max(0, int(sales / np.random.uniform(8, 14)))

            rows.append({
                "Store": store_id, "Date": date, "Sales": sales,
                "Customers": customers, "Open": 1, "Promo": promo,
                "StateHoliday": "0",
                "SchoolHoliday": int(date in school_holidays),
                **meta,
            })

    df = pd.DataFrame(rows)
    print(f"✅ Generated {len(df):,} rows | {n_stores} stores | {start} → {end}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CLEANING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data and engineer time-based features.

    Steps
    -----
    1. Parse dates, sort chronologically
    2. Remove closed-store rows (Open == 0, Sales == 0)
    3. Fill any residual NaNs
    4. Add calendar features: Year, Month, Week, DayOfWeek, Quarter
    5. Add lag features: lag_7, lag_14, lag_28
    6. Add rolling features: rolling_mean_7, rolling_mean_28
    """
    df = df.copy()

    # ── Date parsing ──────────────────────────────────────────────────────────
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(["Store", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Remove truly closed days ───────────────────────────────────────────────
    before = len(df)
    df = df[~((df["Open"] == 0) & (df["Sales"] == 0))]
    print(f"   Removed {before - len(df):,} closed-store rows")

    # ── Missing value imputation ───────────────────────────────────────────────
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include="object").columns
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])

    missing = df.isnull().sum().sum()
    print(f"   Residual NaNs after imputation: {missing}")

    # ── Calendar features ─────────────────────────────────────────────────────
    df["Year"]       = df["Date"].dt.year
    df["Month"]      = df["Date"].dt.month
    df["Week"]       = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"]  = df["Date"].dt.dayofweek          # 0=Mon … 6=Sun
    df["Quarter"]    = df["Date"].dt.quarter
    df["IsWeekend"]  = (df["DayOfWeek"] >= 5).astype(int)

    # ── Lag & rolling features (per store) ───────────────────────────────────
    for lag in [7, 14, 28]:
        df[f"lag_{lag}"] = df.groupby("Store")["Sales"].shift(lag)

    df["rolling_mean_7"]  = df.groupby("Store")["Sales"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )
    df["rolling_mean_28"] = df.groupby("Store")["Sales"].transform(
        lambda x: x.shift(1).rolling(28, min_periods=1).mean()
    )

    df.dropna(subset=["lag_28"], inplace=True)     # drop early rows missing all lags
    df.reset_index(drop=True, inplace=True)

    print(f"✅ Clean dataset: {len(df):,} rows × {df.shape[1]} cols")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3.  RESAMPLING
# ══════════════════════════════════════════════════════════════════════════════
def resample_store(df: pd.DataFrame, store_id: int, freq: str = "W") -> pd.DataFrame:
    """
    Aggregate one store's sales to weekly ('W') or daily ('D') frequency.

    Parameters
    ----------
    df       : Cleaned DataFrame (all stores)
    store_id : Target store
    freq     : 'W' (weekly) or 'D' (daily)

    Returns
    -------
    pd.DataFrame  Indexed by Date with Sales column
    """
    ts = (
        df[df["Store"] == store_id]
        .set_index("Date")["Sales"]
        .resample(freq)
        .sum()
        .rename("Sales")
        .to_frame()
    )
    ts.index.freq = freq
    print(f"   Store {store_id} | freq={freq} | {len(ts)} periods")
    return ts


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TIME-BASED TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
def train_test_split_ts(ts: pd.DataFrame, test_periods: int = 12):
    """
    Chronological train/test split — NO leakage.

    Parameters
    ----------
    ts           : Time-indexed DataFrame
    test_periods : Number of periods held out for evaluation

    Returns
    -------
    train, test : Two DataFrames
    """
    train = ts.iloc[:-test_periods]
    test  = ts.iloc[-test_periods:]
    print(f"   Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} periods)")
    print(f"   Test : {test.index[0].date()}  → {test.index[-1].date()}  ({len(test)} periods)")
    return train, test


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  STEP 1 — DATA PREPARATION")
    print("="*60)

    raw = generate_rossmann_sample(n_stores=5)
    raw.to_csv(DATA_DIR / "rossmann_sample.csv", index=False)
    print(f"   Raw data saved → data/rossmann_sample.csv\n")

    print("Cleaning & feature engineering …")
    clean = clean_and_engineer(raw)
    clean.to_csv(DATA_DIR / "rossmann_clean.csv", index=False)
    print(f"   Clean data saved → data/rossmann_clean.csv\n")

    print("Resampling store 1 to weekly …")
    ts_store1 = resample_store(clean, store_id=1, freq="W")

    print("\nTrain/test split (12-week hold-out) …")
    train, test = train_test_split_ts(ts_store1, test_periods=12)

    print("\n✅ Data preparation complete.")
    print(f"   Weekly time series shape: {ts_store1.shape}")

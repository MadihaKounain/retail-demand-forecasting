"""
src/eda.py
==========
Exploratory Data Analysis for Retail Demand Forecasting.

Covers:
- Sales trend visualization
- Weekly & annual seasonality
- Seasonal decomposition (additive / multiplicative)
- Stationarity check (ADF test + KPSS)
- Autocorrelation (ACF / PACF) for ARIMA order identification
- Store-level comparison heatmap
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                   # non-interactive backend for scripts
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
OUTPUT  = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────────
PALETTE = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED"]
plt.rcParams.update({
    "figure.facecolor": "#F8FAFC",
    "axes.facecolor":   "#F8FAFC",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.family":       "monospace",
    "axes.titlesize":    12,
    "axes.labelsize":    10,
})


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SALES TREND
# ══════════════════════════════════════════════════════════════════════════════
def plot_sales_trend(df: pd.DataFrame, store_id: int = 1):
    """Plot raw daily sales with 28-day rolling average for one store."""
    store_df = df[df["Store"] == store_id].set_index("Date").sort_index()
    daily    = store_df["Sales"].resample("D").sum()
    rolling  = daily.rolling(28, center=True).mean()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(daily.index, daily, alpha=0.25, color=PALETTE[0])
    ax.plot(daily.index, daily,   color=PALETTE[0], lw=0.8, label="Daily Sales")
    ax.plot(rolling.index, rolling, color=PALETTE[2], lw=2,   label="28-day Rolling Avg")
    ax.set_title(f"Store {store_id} — Sales Trend", fontweight="bold")
    ax.set_ylabel("Sales (€)")
    ax.set_xlabel("")
    ax.legend()
    plt.tight_layout()
    path = OUTPUT / "trend_store1.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"   Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SEASONALITY PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
def plot_seasonality(df: pd.DataFrame, store_id: int = 1):
    """Weekly and monthly average sales bar charts."""
    store_df = df[df["Store"] == store_id]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    all_day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Weekly pattern
    weekly     = store_df.groupby("DayOfWeek")["Sales"].mean()
    day_labels = [all_day_names[i] for i in weekly.index]
    axes[0].bar(day_labels, weekly.values, color=PALETTE[0], alpha=0.85, edgecolor="white")
    axes[0].set_title("Average Sales by Day of Week", fontweight="bold")
    axes[0].set_ylabel("Avg Sales (€)")

    # Monthly pattern
    monthly = store_df.groupby("Month")["Sales"].mean()
    axes[1].bar(range(1, 13), monthly.values, color=PALETTE[1], alpha=0.85, edgecolor="white")
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                              "Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=8)
    axes[1].set_title("Average Sales by Month", fontweight="bold")
    axes[1].set_ylabel("Avg Sales (€)")

    plt.tight_layout()
    path = OUTPUT / "seasonality.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"   Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SEASONAL DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════
def plot_decomposition(ts_weekly: pd.DataFrame):
    """Additive seasonal decomposition of weekly series."""
    decomp = seasonal_decompose(ts_weekly["Sales"], model="additive", period=52)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    components = [
        (ts_weekly["Sales"], "Observed"),
        (decomp.trend,       "Trend"),
        (decomp.seasonal,    "Seasonality"),
        (decomp.resid,       "Residuals"),
    ]
    for ax, (data, label) in zip(axes, components):
        ax.plot(data, color=PALETTE[0], lw=1.2)
        ax.set_ylabel(label, fontsize=9)
        if label == "Residuals":
            ax.axhline(0, color="gray", lw=0.7, ls="--")

    axes[0].set_title("Seasonal Decomposition (Additive, Weekly)", fontweight="bold")
    plt.tight_layout()
    path = OUTPUT / "decomposition.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"   Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  STATIONARITY TESTS
# ══════════════════════════════════════════════════════════════════════════════
def stationarity_report(series: pd.Series, name: str = "Sales") -> dict:
    """
    Run ADF (unit root) and KPSS (trend stationarity) tests.
    Returns dict with results for downstream use.
    """
    print(f"\n{'─'*50}")
    print(f"  Stationarity Tests — {name}")
    print(f"{'─'*50}")

    # ── ADF test ──────────────────────────────────────────────────────────────
    adf_result = adfuller(series.dropna(), autolag="AIC")
    adf_stat, adf_p, _, _, adf_crit, _ = adf_result
    print(f"\n  ADF Test (H0: unit root / non-stationary)")
    print(f"    ADF Statistic : {adf_stat:.4f}")
    print(f"    p-value       : {adf_p:.4f}")
    for k, v in adf_crit.items():
        print(f"    Critical ({k}): {v:.4f}")
    adf_stationary = adf_p < 0.05
    print(f"    ➤ Series is {'STATIONARY ✅' if adf_stationary else 'NON-STATIONARY ❌'} (α=0.05)")

    # ── KPSS test ─────────────────────────────────────────────────────────────
    kpss_stat, kpss_p, _, kpss_crit = kpss(series.dropna(), regression="c", nlags="auto")
    print(f"\n  KPSS Test (H0: stationary)")
    print(f"    KPSS Statistic: {kpss_stat:.4f}")
    print(f"    p-value       : {kpss_p:.4f}")
    kpss_stationary = kpss_p >= 0.05
    print(f"    ➤ Series is {'STATIONARY ✅' if kpss_stationary else 'NON-STATIONARY ❌'} (α=0.05)")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n  {'─'*30}")
    if adf_stationary and kpss_stationary:
        verdict = "Stationary — use ARIMA(p,0,q)"
    elif not adf_stationary or not kpss_stationary:
        verdict = "Non-stationary — apply d=1 differencing → ARIMA(p,1,q)"
    else:
        verdict = "Conflicting results — inspect ACF/PACF manually"
    print(f"  VERDICT: {verdict}")
    print(f"  {'─'*30}\n")

    return {
        "adf_stat": adf_stat, "adf_p": adf_p, "adf_stationary": adf_stationary,
        "kpss_stat": kpss_stat, "kpss_p": kpss_p, "kpss_stationary": kpss_stationary,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ACF / PACF  (for ARIMA order selection)
# ══════════════════════════════════════════════════════════════════════════════
def plot_acf_pacf(series: pd.Series, lags: int = 52):
    """Plot ACF and PACF to help select ARIMA p and q orders."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))
    plot_acf(series.dropna(),  lags=lags, ax=ax1, color=PALETTE[0], alpha=0.05)
    plot_pacf(series.dropna(), lags=lags, ax=ax2, color=PALETTE[1], alpha=0.05, method="ywm")
    ax1.set_title("Autocorrelation Function (ACF) — guides MA(q) order", fontweight="bold")
    ax2.set_title("Partial ACF (PACF) — guides AR(p) order", fontweight="bold")
    plt.tight_layout()
    path = OUTPUT / "acf_pacf.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"   Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  STORE COMPARISON HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
def plot_store_heatmap(df: pd.DataFrame):
    """Monthly median sales heatmap across stores."""
    pivot = (
        df[df["Open"] == 1]
        .assign(YM=df["Date"].dt.to_period("M").astype(str))
        .groupby(["Store", "YM"])["Sales"]
        .median()
        .unstack("YM")
    )
    fig, ax = plt.subplots(figsize=(18, 4))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.3,
                linecolor="white", cbar_kws={"label": "Median Sales (€)"})
    ax.set_title("Monthly Median Sales Heatmap — All Stores", fontweight="bold")
    ax.set_ylabel("Store ID")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    path = OUTPUT / "store_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"   Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_prep import generate_rossmann_sample, clean_and_engineer, resample_store

    print("\n" + "="*60)
    print("  STEP 2 — EXPLORATORY DATA ANALYSIS")
    print("="*60)

    DATA_CSV = ROOT / "data" / "rossmann_clean.csv"
    if DATA_CSV.exists():
        df_clean = pd.read_csv(DATA_CSV, parse_dates=["Date"])
        print(f"   Loaded clean data: {len(df_clean):,} rows")
    else:
        raw      = generate_rossmann_sample()
        df_clean = clean_and_engineer(raw)

    print("\n[1] Sales trend …")
    plot_sales_trend(df_clean, store_id=1)

    print("[2] Seasonality patterns …")
    plot_seasonality(df_clean, store_id=1)

    ts_weekly = resample_store(df_clean, store_id=1, freq="W")

    print("[3] Seasonal decomposition …")
    plot_decomposition(ts_weekly)

    print("[4] Stationarity tests …")
    stats = stationarity_report(ts_weekly["Sales"], name="Weekly Sales (Store 1)")

    print("[5] ACF / PACF …")
    if not stats["adf_stationary"]:
        plot_acf_pacf(ts_weekly["Sales"].diff().dropna(), lags=40)
    else:
        plot_acf_pacf(ts_weekly["Sales"], lags=40)

    print("[6] Store heatmap …")
    plot_store_heatmap(df_clean)

    print("\n✅ EDA complete. All plots saved to outputs/")

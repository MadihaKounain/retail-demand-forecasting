"""
src/modeling.py
===============
Demand Forecasting Models:
  1. SARIMA  — statsmodels SARIMAX
  2. Prophet — Meta's open-source forecasting library

Both models follow a strict time-based train/test split (no leakage).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Graceful Prophet import
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️  Prophet not installed. Run: pip install prophet")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

PALETTE = ["#2563EB", "#DC2626", "#16A34A", "#D97706"]
plt.rcParams.update({
    "figure.facecolor": "#F8FAFC",
    "axes.facecolor":   "#F8FAFC",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.family":       "monospace",
})


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════
def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error (excludes zero-actual rows)."""
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — SARIMA
# ══════════════════════════════════════════════════════════════════════════════
class SARIMAForecaster:
    """
    Seasonal ARIMA forecaster using statsmodels SARIMAX.

    Default order: SARIMA(1,1,1)(1,1,1)[52]
    - (p,d,q)     = (1,1,1)  — AR lag, differencing, MA lag
    - (P,D,Q,s)   = (1,1,1,52) — seasonal AR, differencing, MA over 52-week cycle

    To find optimal orders, use auto_arima from pmdarima (not required here).
    """

    def __init__(self, order=(1,1,1), seasonal_order=(1,1,1,52)):
        self.order          = order
        self.seasonal_order = seasonal_order
        self.model          = None
        self.result         = None
        self.train          = None
        self.test           = None

    def fit(self, train: pd.DataFrame):
        """Fit SARIMA on training data."""
        self.train = train
        print(f"\n  Fitting SARIMA{self.order}×{self.seasonal_order} …")
        self.model  = SARIMAX(
            train["Sales"],
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.result = self.model.fit(disp=False, maxiter=200)
        print(f"  AIC: {self.result.aic:.2f}  |  BIC: {self.result.bic:.2f}")
        return self

    def predict(self, steps: int) -> pd.Series:
        """Forecast `steps` periods ahead."""
        forecast = self.result.get_forecast(steps=steps)
        pred     = forecast.predicted_mean
        ci       = forecast.conf_int(alpha=0.10)    # 90% confidence interval
        self._pred_mean = pred
        self._pred_ci   = ci
        return pred

    def evaluate(self, test: pd.DataFrame) -> dict:
        """Predict on test set and return metrics."""
        self.test   = test
        pred        = self.predict(len(test))
        pred.index  = test.index

        actual    = test["Sales"].values
        predicted = np.maximum(pred.values, 0)

        metrics = {
            "model":  "SARIMA",
            "MAPE":   round(mape(actual, predicted), 2),
            "RMSE":   round(rmse(actual, predicted), 2),
            "MAE":    round(mae(actual, predicted), 2),
        }
        print(f"\n  ── SARIMA Evaluation ──")
        print(f"  MAPE : {metrics['MAPE']:.2f}%")
        print(f"  RMSE : {metrics['RMSE']:,.0f}")
        print(f"  MAE  : {metrics['MAE']:,.0f}")
        return metrics

    def plot_forecast(self, test: pd.DataFrame, title: str = "SARIMA Forecast"):
        pred  = self._pred_mean.copy()
        pred.index = test.index

        fig, ax = plt.subplots(figsize=(14, 5))

        # Training tail (last 52 weeks)
        tail = self.train.tail(52)
        ax.plot(tail.index, tail["Sales"], color=PALETTE[0], lw=1.4, label="Train (last 52w)")

        # Test actuals
        ax.plot(test.index, test["Sales"], color="black", lw=1.8, ls="--", label="Actual")

        # Forecast
        ax.plot(pred.index, pred, color=PALETTE[1], lw=2, label="SARIMA Forecast")

        # Confidence interval
        if hasattr(self, "_pred_ci"):
            ci = self._pred_ci.copy()
            ci.index = test.index
            ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
                            color=PALETTE[1], alpha=0.15, label="90% CI")

        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Weekly Sales (€)")
        ax.legend()
        plt.tight_layout()
        path = OUTPUT / "sarima_forecast.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"   Saved → {path}")

    def plot_diagnostics(self):
        """Ljung-Box, residual ACF, histogram."""
        try:
            fig = self.result.plot_diagnostics(figsize=(14, 8))
            fig.suptitle("SARIMA Model Diagnostics", fontweight="bold", y=1.02)
            plt.tight_layout()
            path = OUTPUT / "sarima_diagnostics.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"   Saved → {path}")
        except ValueError as e:
            print(f"   Diagnostics skipped (series too short for lags): {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — PROPHET
# ══════════════════════════════════════════════════════════════════════════════
class ProphetForecaster:
    """
    Facebook Prophet forecaster with weekly + yearly seasonality.
    Prophet requires a DataFrame with columns [ds, y].
    """

    def __init__(self,
                 changepoint_prior_scale: float = 0.15,
                 seasonality_prior_scale: float = 10.0):
        if not PROPHET_AVAILABLE:
            raise ImportError("Install prophet: pip install prophet")
        self.changepoint_prior = changepoint_prior_scale
        self.seasonality_prior = seasonality_prior_scale
        self.model   = None
        self.train   = None

    @staticmethod
    def _to_prophet_df(ts: pd.DataFrame) -> pd.DataFrame:
        return ts.reset_index().rename(columns={"Date": "ds", "Sales": "y"})

    def fit(self, train: pd.DataFrame):
        """Fit Prophet on training data."""
        self.train = train
        df_prophet = self._to_prophet_df(train)

        print(f"\n  Fitting Prophet …")
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior,
            seasonality_prior_scale=self.seasonality_prior,
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.90,
        )
        self.model.fit(df_prophet)
        print("  Prophet training complete.")
        return self

    def predict(self, steps: int) -> pd.DataFrame:
        """Forecast `steps` periods ahead and return full Prophet DataFrame."""
        future     = self.model.make_future_dataframe(periods=steps, freq="W")
        self._forecast_df = self.model.predict(future)
        return self._forecast_df

    def evaluate(self, test: pd.DataFrame) -> dict:
        """Return metrics on the test window."""
        forecast_df = self.predict(len(test))
        pred_rows   = forecast_df.tail(len(test))

        actual    = test["Sales"].values
        predicted = np.maximum(pred_rows["yhat"].values, 0)

        metrics = {
            "model": "Prophet",
            "MAPE":  round(mape(actual, predicted), 2),
            "RMSE":  round(rmse(actual, predicted), 2),
            "MAE":   round(mae(actual, predicted), 2),
        }
        print(f"\n  ── Prophet Evaluation ──")
        print(f"  MAPE : {metrics['MAPE']:.2f}%")
        print(f"  RMSE : {metrics['RMSE']:,.0f}")
        print(f"  MAE  : {metrics['MAE']:,.0f}")
        return metrics

    def plot_forecast(self, test: pd.DataFrame):
        """Plot Prophet forecast vs actuals."""
        fdf    = self._forecast_df
        tail   = self.train.tail(52)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(tail.index, tail["Sales"], color=PALETTE[0], lw=1.4, label="Train (last 52w)")
        ax.plot(test.index, test["Sales"], color="black",    lw=1.8, ls="--", label="Actual")

        pred_rows = fdf.tail(len(test))
        pred_dates = pd.to_datetime(pred_rows["ds"].values)
        ax.plot(pred_dates, pred_rows["yhat"].values,      color=PALETTE[2], lw=2, label="Prophet Forecast")
        ax.fill_between(pred_dates, pred_rows["yhat_lower"], pred_rows["yhat_upper"],
                        color=PALETTE[2], alpha=0.15, label="90% CI")

        ax.set_title("Prophet Forecast vs Actuals", fontweight="bold")
        ax.set_ylabel("Weekly Sales (€)")
        ax.legend()
        plt.tight_layout()
        path = OUTPUT / "prophet_forecast.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"   Saved → {path}")

    def plot_components(self):
        """Prophet's built-in component plot (trend + seasonalities)."""
        fig = self.model.plot_components(self._forecast_df, figsize=(14, 8))
        fig.suptitle("Prophet Forecast Components", fontweight="bold", y=1.01)
        plt.tight_layout()
        path = OUTPUT / "prophet_components.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def compare_models(metrics_list: list[dict]) -> pd.DataFrame:
    """Print and save a side-by-side metrics comparison table."""
    df = pd.DataFrame(metrics_list).set_index("model")
    print("\n" + "═"*40)
    print("  MODEL COMPARISON")
    print("═"*40)
    print(df.to_string())
    print("═"*40)
    path = OUTPUT / "metrics_summary.csv"
    df.to_csv(path)
    print(f"\n  Metrics saved → {path}")
    return df


def plot_model_comparison(sarima_metrics: dict, prophet_metrics: dict = None):
    """Bar chart comparing MAPE and RMSE across models."""
    models  = [sarima_metrics]
    if prophet_metrics:
        models.append(prophet_metrics)

    names  = [m["model"] for m in models]
    mapes  = [m["MAPE"]  for m in models]
    rmses  = [m["RMSE"]  for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    colors = PALETTE[:len(names)]
    ax1.bar(names, mapes, color=colors, edgecolor="white", width=0.5)
    ax1.set_title("MAPE (%) — lower is better", fontweight="bold")
    ax1.set_ylabel("MAPE (%)")
    for i, v in enumerate(mapes):
        ax1.text(i, v + 0.2, f"{v:.1f}%", ha="center", fontweight="bold")

    ax2.bar(names, rmses, color=colors, edgecolor="white", width=0.5)
    ax2.set_title("RMSE — lower is better", fontweight="bold")
    ax2.set_ylabel("RMSE (€)")
    for i, v in enumerate(rmses):
        ax2.text(i, v + 50, f"{v:,.0f}", ha="center", fontweight="bold")

    plt.suptitle("Model Comparison", fontweight="bold", fontsize=13)
    plt.tight_layout()
    path = OUTPUT / "model_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"   Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_prep import (
        generate_rossmann_sample, clean_and_engineer,
        resample_store, train_test_split_ts
    )

    print("\n" + "="*60)
    print("  STEP 3 — MODELING")
    print("="*60)

    DATA_CSV = ROOT / "data" / "rossmann_clean.csv"
    if DATA_CSV.exists():
        df_clean = pd.read_csv(DATA_CSV, parse_dates=["Date"])
    else:
        raw      = generate_rossmann_sample()
        df_clean = clean_and_engineer(raw)

    ts = resample_store(df_clean, store_id=1, freq="W")
    train, test = train_test_split_ts(ts, test_periods=12)

    # ── SARIMA ────────────────────────────────────────────────────────────────
    sarima = SARIMAForecaster(order=(1,1,1), seasonal_order=(1,1,1,52))
    sarima.fit(train)
    sarima_metrics = sarima.evaluate(test)
    sarima.plot_forecast(test)
    sarima.plot_diagnostics()

    all_metrics = [sarima_metrics]

    # ── Prophet (optional) ────────────────────────────────────────────────────
    if PROPHET_AVAILABLE:
        prophet = ProphetForecaster()
        prophet.fit(train)
        prophet_metrics = prophet.evaluate(test)
        prophet.plot_forecast(test)
        prophet.plot_components()
        all_metrics.append(prophet_metrics)
    else:
        prophet_metrics = None
        print("\n  Skipping Prophet (not installed)")

    # ── Comparison ────────────────────────────────────────────────────────────
    compare_models(all_metrics)
    plot_model_comparison(sarima_metrics, prophet_metrics)

    print("\n✅ Modeling complete.")

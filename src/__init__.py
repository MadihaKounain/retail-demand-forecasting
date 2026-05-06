# demand_forecasting/src/__init__.py
from .data_prep import generate_rossmann_sample, clean_and_engineer, resample_store, train_test_split_ts
from .eda import stationarity_report
from .modeling import SARIMAForecaster, PROPHET_AVAILABLE
from .evaluate import inventory_recommendations, print_business_insights

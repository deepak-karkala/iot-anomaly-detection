'''
- Verify feature calculation logic for a single inference date,
	including correct handling of lookback periods for lags/rolling windows.
	Ensure output schema matches what the inference script expects.
'''

from datetime import date, datetime, timedelta

import pandas as pd
import pytest
# Assuming the script is saved as 'feature_engineering_inference.py'
# Adjust the import path if necessary
from feature_engineering_inference import calculate_inference_features
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import (DateType, DoubleType, StringType, StructField,
                               StructType, TimestampType)


@pytest.fixture(scope="session")
def spark():
    """Pytest fixture to create a SparkSession for testing."""
    return SparkSession.builder \
        .master("local[2]") \
        .appName("InferenceFeatureEngTest") \
        .config("spark.sql.session.timeZone", "UTC") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()

# --- Schemas for Mock Data (Should match previous schemas) ---
METER_INPUT_SCHEMA = StructType([
    StructField("event_ts", TimestampType(), True),
    StructField("meter_id", StringType(), True),
    StructField("apartment_id", StringType(), True),
    StructField("building_id", StringType(), True),
    StructField("energy_kwh", DoubleType(), True),
    StructField("room_temp_c", DoubleType(), True),
    StructField("setpoint_temp_c", DoubleType(), True),
    StructField("hot_water_litres", DoubleType(), True),
])

WEATHER_INPUT_SCHEMA = StructType([
    StructField("date_col_name", StringType(), True), # Date as string
    StructField("hdd", DoubleType(), True),
    StructField("avg_temp_c", DoubleType(), True),
    # StructField("building_id_weather", StringType(), True), # If needed
])

# Expected output columns for inference (subset of features potentially)
EXPECTED_INFERENCE_FEATURE_COLS = [
     "apartment_id", "building_id", "event_date",
     "daily_energy_kwh", "avg_temp_diff", "hdd", "avg_temp_c",
     "energy_lag_1d", "energy_roll_avg_7d", "temp_diff_lag_1d"
]


# --- Test Cases ---

def test_calc_inference_features_success(spark):
    """Test feature calculation for a specific inference date."""
    inference_date_str = "2024-01-17"
    lookback_days = 7
    start_date_dt = datetime.strptime(inference_date_str, '%Y-%m-%d').date() - timedelta(days=lookback_days)

    # Provide data covering lookback + inference date
    meter_data = [
        # Day 1 (lookback)
        (datetime(2024, 1, 15, 0, 0, 0), "m1", "apt1", "bldgA", 100.0, 20.0, 21.0, 50.0),
        (datetime(2024, 1, 15, 23, 59, 0), "m1", "apt1", "bldgA", 110.0, 20.5, 21.0, 55.0), # E=10
         # Day 2 (lookback)
        (datetime(2024, 1, 16, 0, 0, 0), "m1", "apt1", "bldgA", 110.0, 20.6, 21.0, 55.0),
        (datetime(2024, 1, 16, 23, 59, 0), "m1", "apt1", "bldgA", 122.0, 20.8, 21.0, 60.0), # E=12
        # Day 3 (inference date)
        (datetime(2024, 1, 17, 0, 0, 0), "m1", "apt1", "bldgA", 122.0, 20.7, 21.0, 60.0),
        (datetime(2024, 1, 17, 23, 59, 0), "m1", "apt1", "bldgA", 130.0, 20.7, 21.0, 62.0), # E=8
        # Apt 2 (inference date)
        (datetime(2024, 1, 17, 0, 0, 0), "m2", "apt2", "bldgA", 50.0, 21.0, 22.0, 10.0),
        (datetime(2024, 1, 17, 23, 59, 0), "m2", "apt2", "bldgA", 59.0, 21.5, 22.0, 12.0), # E=9
    ]
    meter_df = spark.createDataFrame(meter_data, METER_INPUT_SCHEMA)

    weather_data = [
        ("2024-01-15", 10.0, 5.0),
        ("2024-01-16", 12.0, 3.0),
        ("2024-01-17", 11.0, 4.0),
    ]
    weather_df = spark.createDataFrame(weather_data, WEATHER_INPUT_SCHEMA)

    # Mock spark reads
    dummy_meter_path = "dummy/meter/inf"
    dummy_weather_path = "dummy/weather/inf"
    spark.read.format("parquet").option("path", dummy_meter_path).load = lambda: meter_df
    spark.read.format("parquet").option("path", dummy_weather_path).load = lambda: weather_df

    output_df = calculate_inference_features(spark, dummy_meter_path, dummy_weather_path, inference_date_str)

    # Assertions
    assert output_df is not None
    assert output_df.count() == 2 # Only records for the inference date (apt1, apt2)
    assert sorted(output_df.columns) == sorted(EXPECTED_INFERENCE_FEATURE_COLS)

    output_pd = output_df.orderBy("apartment_id").toPandas()

    # Check Apt 1 (index 0)
    assert output_pd.loc[0, "apartment_id"] == "apt1"
    assert output_pd.loc[0, "event_date"] == date(2024, 1, 17)
    assert output_pd.loc[0, "daily_energy_kwh"] == pytest.approx(8.0)
    assert output_pd.loc[0, "hdd"] == 11.0
    assert output_pd.loc[0, "energy_lag_1d"] == pytest.approx(12.0) # Energy from 16th
    # Rolling avg = avg(10, 12, 8) = 10
    assert output_pd.loc[0, "energy_roll_avg_7d"] == pytest.approx(10.0)

    # Check Apt 2 (index 1)
    assert output_pd.loc[1, "apartment_id"] == "apt2"
    assert output_pd.loc[1, "event_date"] == date(2024, 1, 17)
    assert output_pd.loc[1, "daily_energy_kwh"] == pytest.approx(9.0)
    assert pd.isna(output_pd.loc[1, "energy_lag_1d"]) # No prior day data for Apt 2
    assert output_pd.loc[1, "energy_roll_avg_7d"] == pytest.approx(9.0) # Rolling avg is just itself


def test_calc_inference_features_no_data_for_date(spark):
    """Tests when no meter data exists for the target inference date."""
    inference_date_str = "2024-01-17"
    # Provide data only for lookback period
    meter_data = [
        (datetime(2024, 1, 15, 0, 0, 0), "m1", "apt1", "bldgA", 100.0, 20.0, 21.0, 50.0),
        (datetime(2024, 1, 15, 23, 59, 0), "m1", "apt1", "bldgA", 110.0, 20.5, 21.0, 55.0),
        (datetime(2024, 1, 16, 0, 0, 0), "m1", "apt1", "bldgA", 110.0, 20.6, 21.0, 55.0),
        (datetime(2024, 1, 16, 23, 59, 0), "m1", "apt1", "bldgA", 122.0, 20.8, 21.0, 60.0),
    ]
    meter_df = spark.createDataFrame(meter_data, METER_INPUT_SCHEMA)
    weather_df = spark.createDataFrame([("2024-01-17", 11.0, 4.0)], WEATHER_INPUT_SCHEMA) # Weather exists

    dummy_meter_path = "dummy/meter/inf/nodate"
    dummy_weather_path = "dummy/weather/inf/nodate"
    spark.read.format("parquet").option("path", dummy_meter_path).load = lambda: meter_df
    spark.read.format("parquet").option("path", dummy_weather_path).load = lambda: weather_df

    output_df = calculate_inference_features(spark, dummy_meter_path, dummy_weather_path, inference_date_str)

    assert output_df is None # Function returns None if no data for target date

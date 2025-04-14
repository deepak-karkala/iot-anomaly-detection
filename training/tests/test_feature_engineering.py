from datetime import date, datetime

import pandas as pd  # Needed for comparison involving NaNs
import pytest
# Assuming the script is saved as 'feature_engineering.py'
# Adjust the import path if necessary
from feature_engineering import \
    calculate_features  # Assuming the function is directly importable
from pyspark.sql import SparkSession
from pyspark.sql.functions import col  # Import col explicitly
from pyspark.sql.types import (DateType, DoubleType, IntegerType, StringType,
                               StructField, StructType, TimestampType)


@pytest.fixture(scope="session")
def spark():
    """Pytest fixture to create a SparkSession for testing."""
    return SparkSession.builder \
        .master("local[2]") \
        .appName("FeatureEngineeringTest") \
        .config("spark.sql.session.timeZone", "UTC") \
        .config("spark.sql.shuffle.partitions", "2") # Keep partitions low for local testing
        .getOrCreate()

# --- Schemas for Mock Data ---
METER_INPUT_SCHEMA = StructType([
    StructField("event_ts", TimestampType(), True),
    StructField("meter_id", StringType(), True),
    StructField("apartment_id", StringType(), True),
    StructField("building_id", StringType(), True),
    StructField("energy_kwh", DoubleType(), True),
    StructField("room_temp_c", DoubleType(), True),
    StructField("setpoint_temp_c", DoubleType(), True),
    StructField("hot_water_litres", DoubleType(), True),
    # Assuming date partitions were handled by the previous step
    # StructField("year", IntegerType(), True),
    # StructField("month", IntegerType(), True),
    # StructField("day", IntegerType(), True),
])

WEATHER_INPUT_SCHEMA = StructType([
    StructField("date_col_name", StringType(), True), # Date as string initially
    StructField("hdd", DoubleType(), True),
    StructField("avg_temp_c", DoubleType(), True),
    StructField("building_id_weather", StringType(), True), # Optional building ID if weather is specific
])

# Expected Schema after calculate_features (must match Feature Store Def)
EXPECTED_FEATURE_SCHEMA = StructType([
    StructField("apartment_record_id", StringType(), False), # Assuming False=NotNull post-dropna
    StructField("event_time", TimestampType(), False), # Assuming False=NotNull post-dropna
    StructField("event_date", DateType(), True),
    StructField("building_id", StringType(), True),
    StructField("daily_energy_kwh", DoubleType(), True),
    StructField("avg_temp_diff", DoubleType(), True),
    StructField("daily_water_l", DoubleType(), True),
    StructField("hdd", DoubleType(), True),
    StructField("avg_temp_c", DoubleType(), True),
    StructField("energy_lag_1d", DoubleType(), True),
    StructField("energy_roll_avg_7d", DoubleType(), True),
    StructField("temp_diff_lag_1d", DoubleType(), True),
])

# --- Test Cases ---

def test_calculate_features_happy_path(spark):
    """Tests feature calculation with valid inputs."""
    meter_data = [
        # Apt 1, Day 1 (2 readings)
        (datetime(2024, 1, 15, 0, 0, 0), "m1", "apt1", "bldgA", 100.0, 20.0, 21.0, 50.0),
        (datetime(2024, 1, 15, 23, 59, 0), "m1", "apt1", "bldgA", 110.0, 20.5, 21.0, 55.0), # Energy=10, Water=5
        # Apt 1, Day 2
        (datetime(2024, 1, 16, 0, 0, 0), "m1", "apt1", "bldgA", 110.0, 20.6, 21.0, 55.0),
        (datetime(2024, 1, 16, 23, 59, 0), "m1", "apt1", "bldgA", 122.0, 20.8, 21.0, 60.0), # Energy=12, Water=5
        # Apt 1, Day 3
        (datetime(2024, 1, 17, 0, 0, 0), "m1", "apt1", "bldgA", 122.0, 20.7, 21.0, 60.0),
        (datetime(2024, 1, 17, 23, 59, 0), "m1", "apt1", "bldgA", 130.0, 20.7, 21.0, 62.0), # Energy=8, Water=2
        # Apt 2, Day 1
        (datetime(2024, 1, 15, 0, 0, 0), "m2", "apt2", "bldgA", 50.0, 21.0, 22.0, 10.0),
        (datetime(2024, 1, 15, 23, 59, 0), "m2", "apt2", "bldgA", 55.0, 21.5, 22.0, 12.0), # Energy=5, Water=2
    ]
    meter_df = spark.createDataFrame(meter_data, METER_INPUT_SCHEMA)

    weather_data = [
        ("2024-01-15", 10.0, 5.0, "bldgA"),
        ("2024-01-16", 12.0, 3.0, "bldgA"),
        ("2024-01-17", 11.0, 4.0, "bldgA"),
        ("2024-01-18", 9.0, 6.0, "bldgA"), # Extra day
    ]
    weather_df = spark.createDataFrame(weather_data, WEATHER_INPUT_SCHEMA)

    # Create dummy S3 paths (content doesn't matter for this test)
    dummy_meter_path = "dummy/meter/path"
    dummy_weather_path = "dummy/weather/path"
    # Mock spark.read.load to return our test dataframes
    spark.read.format("parquet").option("path", dummy_meter_path).load = lambda: meter_df
    spark.read.format("parquet").option("path", dummy_weather_path).load = lambda: weather_df


    output_df = calculate_features(spark, dummy_meter_path, dummy_weather_path, "2024-01-15", "2024-01-17")

    # Assertions
    assert output_df is not None
    assert output_df.count() == 4 # 3 days for apt1, 1 day for apt2
    # Check schema loosely first, then compare field names/types if needed
    assert len(output_df.schema.fields) == len(EXPECTED_FEATURE_SCHEMA.fields)
    # Could do a more rigorous schema comparison if desired

    output_pd = output_df.orderBy("apartment_record_id", "event_date").toPandas()

    # Check Apt 1, Day 2 (index 1)
    assert output_pd.loc[1, "apartment_record_id"] == "apt1"
    assert output_pd.loc[1, "event_date"] == date(2024, 1, 16)
    assert output_pd.loc[1, "daily_energy_kwh"] == pytest.approx(12.0)
    assert output_pd.loc[1, "daily_water_l"] == pytest.approx(5.0)
    assert output_pd.loc[1, "avg_temp_diff"] == pytest.approx(20.7 - 21.0) # Avg(20.6, 20.8) - 21.0
    assert output_pd.loc[1, "hdd"] == 12.0
    assert output_pd.loc[1, "energy_lag_1d"] == pytest.approx(10.0) # Energy from Day 1
    # Rolling avg for day 2 = avg(day1, day2) = avg(10, 12) = 11
    assert output_pd.loc[1, "energy_roll_avg_7d"] == pytest.approx(11.0)

    # Check Apt 1, Day 3 (index 2)
    assert output_pd.loc[2, "energy_lag_1d"] == pytest.approx(12.0) # Energy from Day 2
    # Rolling avg for day 3 = avg(day1, day2, day3) = avg(10, 12, 8) = 10
    assert output_pd.loc[2, "energy_roll_avg_7d"] == pytest.approx(10.0)

    # Check Apt 2, Day 1 (index 3) - lag features should be null/NaN
    assert output_pd.loc[3, "apartment_record_id"] == "apt2"
    assert output_pd.loc[3, "event_date"] == date(2024, 1, 15)
    assert pd.isna(output_pd.loc[3, "energy_lag_1d"])
    assert pd.isna(output_pd.loc[3, "temp_diff_lag_1d"])
    assert output_pd.loc[3, "energy_roll_avg_7d"] == pytest.approx(5.0) # Rolling avg of just day 1 = 5

def test_calculate_features_no_meter_data(spark):
    """Tests when no meter data exists for the date range."""
    meter_df = spark.createDataFrame([], METER_INPUT_SCHEMA)
    # Weather data doesn't matter if meter data is empty
    weather_df = spark.createDataFrame([("2024-01-15", 10.0, 5.0, "bldgA")], WEATHER_INPUT_SCHEMA)

    dummy_meter_path = "dummy/meter/path/empty"
    dummy_weather_path = "dummy/weather/path/empty"
    spark.read.format("parquet").option("path", dummy_meter_path).load = lambda: meter_df
    spark.read.format("parquet").option("path", dummy_weather_path).load = lambda: weather_df

    output_df = calculate_features(spark, dummy_meter_path, dummy_weather_path, "2024-01-15", "2024-01-17")

    assert output_df is None # Function should return None or empty


def test_calculate_features_missing_weather(spark):
    """Tests when weather data is missing for a specific day."""
    meter_data = [
        (datetime(2024, 1, 15, 0, 0, 0), "m1", "apt1", "bldgA", 100.0, 20.0, 21.0, 50.0),
        (datetime(2024, 1, 15, 23, 59, 0), "m1", "apt1", "bldgA", 110.0, 20.5, 21.0, 55.0),
    ]
    meter_df = spark.createDataFrame(meter_data, METER_INPUT_SCHEMA)

    weather_data = [ # No data for 2024-01-15
        ("2024-01-16", 12.0, 3.0, "bldgA"),
    ]
    weather_df = spark.createDataFrame(weather_data, WEATHER_INPUT_SCHEMA)

    dummy_meter_path = "dummy/meter/path/mw"
    dummy_weather_path = "dummy/weather/path/mw"
    spark.read.format("parquet").option("path", dummy_meter_path).load = lambda: meter_df
    spark.read.format("parquet").option("path", dummy_weather_path).load = lambda: weather_df

    output_df = calculate_features(spark, dummy_meter_path, dummy_weather_path, "2024-01-15", "2024-01-15")

    assert output_df.count() == 1
    output_pd = output_df.toPandas()
    # Weather features should be null due to left join
    assert pd.isna(output_pd.loc[0, "hdd"])
    assert pd.isna(output_pd.loc[0, "avg_temp_c"])
    # Meter-derived features should still be calculated
    assert output_pd.loc[0, "daily_energy_kwh"] == pytest.approx(10.0)

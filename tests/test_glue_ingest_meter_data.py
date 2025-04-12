'''
- Test Setup (conftest.py or fixture): 
    The spark() fixture provides a local SparkSession needed to create and
    manipulate DataFrames during testing. You might need to install pytest
    and pyspark. If you use pytest-spark, it handles the session setup automatically.

- Schemas:
    Define RAW_INPUT_SCHEMA and EXPECTED_OUTPUT_SCHEMA clearly. This makes tests
    easier to read and ensures you're checking the final structure correctly.

- Test Cases:
    - test_successful_transformation:
        Checks the "happy path" with valid data, ensuring correct schema, count,
        partition values, and timestamp conversion.
    - test_dropna_null_key_columns: 
        Verifies that rows with nulls in meter_id orinvalid/null timestamps
        (resulting in null event_ts) are dropped.
    - test_empty_input:
        Ensures the function handles an empty input DataFrame gracefully.
    - test_incorrect_timestamp_format:
        Checks that if the input format doesn't match the expected format,
        to_timestamp yields null, leading to the row being dropped by dropna.
    - test_missing_input_column:
        Checks that the function raises a ValueError early if a column required
        for transformation is absent.
    - --- Add more tests for edge cases as needed ---
    -   # - Test different valid timestamp formats if the source can vary
        # - Test null values in non-key columns (should they be kept or imputed?)
        # - Test data type mismatches if input isn't strictly typed JSON

- Running Tests:
    - Install dependencies: pip install pytest pyspark (or pytest-spark)
    - Save the Glue script and the test script (e.g., test_glue_ingest_meter_data.py)
        in your project structure.
    - Run pytest from your terminal in the directory containing the test file: pytest
'''


from datetime import datetime

import pytest
from glue_ingest_meter_data import transform_meter_data
from pyspark.sql import SparkSession
from pyspark.sql.types import (DoubleType, IntegerType, StringType,
                               StructField, StructType, TimestampType)


@pytest.fixture(scope="session")
def spark():
    """Pytest fixture to create a SparkSession for testing."""
    return SparkSession.builder \
        .master("local[2]") \
        .appName("GlueUnitTest") \
        .config("spark.sql.session.timeZone", "UTC") \
        .getOrCreate()

# Define the schema for the raw input data used in tests
# Must include all columns selected in the transform function
RAW_INPUT_SCHEMA = StructType([
    StructField("timestamp_str", StringType(), True),
    StructField("meter_id", StringType(), True),
    StructField("apartment_id", StringType(), True),
    StructField("building_id", StringType(), True),
    StructField("energy_kwh", DoubleType(), True),
    StructField("room_temp_c", DoubleType(), True),
    StructField("setpoint_temp_c", DoubleType(), True),
    StructField("hot_water_litres", DoubleType(), True),
    StructField("extra_raw_col", StringType(), True) # Include extra columns that should be dropped
])

# Define the expected schema AFTER transformation
EXPECTED_OUTPUT_SCHEMA = StructType([
    StructField("event_ts", TimestampType(), True),
    StructField("meter_id", StringType(), True),
    StructField("apartment_id", StringType(), True),
    StructField("building_id", StringType(), True),
    StructField("energy_kwh", DoubleType(), True),
    StructField("room_temp_c", DoubleType(), True),
    StructField("setpoint_temp_c", DoubleType(), True),
    StructField("hot_water_litres", DoubleType(), True),
    StructField("year", IntegerType(), True),
    StructField("month", IntegerType(), True),
    StructField("day", IntegerType(), True),
])

# --- Test Cases ---

def test_successful_transformation(spark):
    """Tests standard, valid input data transformation."""
    input_data = [
        ("2024-01-15 10:30:00", "meter_001", "apt_1", "bldg_A", 10.5, 21.5, 22.0, 5.0, "more_data"),
        ("2024-01-15 10:40:00", "meter_002", "apt_2", "bldg_A", 1.2, 20.0, 21.0, 0.0, "more_data"),
        ("2024-01-16 08:00:00", "meter_001", "apt_1", "bldg_A", 11.0, 21.7, 22.0, 15.0, "data"),
    ]
    input_df = spark.createDataFrame(input_data, RAW_INPUT_SCHEMA)

    output_df = transform_meter_data(spark, input_df)

    # Assertions
    assert output_df.count() == 3
    assert output_df.schema == EXPECTED_OUTPUT_SCHEMA # Check exact schema
    # Check partition columns
    first_row = output_df.orderBy("event_ts").first()
    assert first_row["year"] == 2024
    assert first_row["month"] == 1
    assert first_row["day"] == 15
    assert first_row["event_ts"] == datetime(2024, 1, 15, 10, 30, 0)
    # Check a value
    assert first_row["room_temp_c"] == 21.5
    # Check that extra column is dropped
    assert "extra_raw_col" not in output_df.columns


def test_dropna_null_key_columns(spark):
    """Tests that rows with null meter_id or event_ts (after conversion) are dropped."""
    input_data = [
        ("2024-01-15 10:30:00", "meter_001", "apt_1", "bldg_A", 10.5, 21.5, 22.0, 5.0, "valid"),
        ("2024-01-15 10:40:00", None,        "apt_2", "bldg_A", 1.2, 20.0, 21.0, 0.0, "null meter_id"), # Should be dropped
        ("invalid-timestamp",   "meter_003", "apt_3", "bldg_B", 5.0, 20.5, 21.0, 2.0, "invalid ts"),  # Should be dropped
        (None,                  "meter_004", "apt_4", "bldg_B", 6.0, 20.6, 21.0, 3.0, "null ts str"), # Should be dropped
    ]
    input_df = spark.createDataFrame(input_data, RAW_INPUT_SCHEMA)

    output_df = transform_meter_data(spark, input_df)

    # Assertions
    assert output_df.count() == 1 # Only the first row should remain
    remaining_row = output_df.first()
    assert remaining_row["meter_id"] == "meter_001"


def test_empty_input(spark):
    """Tests behavior with an empty input DataFrame."""
    input_df = spark.createDataFrame([], RAW_INPUT_SCHEMA)
    output_df = transform_meter_data(spark, input_df)

    # Assertions
    assert output_df.count() == 0
    assert output_df.schema == EXPECTED_OUTPUT_SCHEMA # Schema should still match


def test_incorrect_timestamp_format(spark):
    """Tests that incorrect timestamp format leads to null event_ts and row drop."""
    input_data = [
        # Uses MM/dd/yyyy format, but script expects yyyy-MM-dd
        ("01/15/2024 10:30:00", "meter_005", "apt_5", "bldg_C", 10.5, 21.5, 22.0, 5.0, "wrong_fmt"),
    ]
    input_df = spark.createDataFrame(input_data, RAW_INPUT_SCHEMA)

    # Pass the default format used in the main script logic
    output_df = transform_meter_data(spark, input_df)

    # Assertions
    # Since event_ts will be null due to format mismatch, dropna should remove it
    assert output_df.count() == 0


def test_missing_input_column(spark):
    """Tests that the function raises an error if a required column is missing."""
    # Create data missing 'room_temp_c'
    partial_schema = StructType([
        StructField("timestamp_str", StringType(), True),
        StructField("meter_id", StringType(), True),
        StructField("apartment_id", StringType(), True),
        StructField("building_id", StringType(), True),
        StructField("energy_kwh", DoubleType(), True),
        # Missing room_temp_c
        StructField("setpoint_temp_c", DoubleType(), True),
        StructField("hot_water_litres", DoubleType(), True),
        StructField("extra_raw_col", StringType(), True)
    ])
    input_data = [
        ("2024-01-15 10:30:00", "meter_001", "apt_1", "bldg_A", 10.5, 22.0, 5.0, "ignore_me"),
    ]
    input_df = spark.createDataFrame(input_data, partial_schema)

    # Assert that a ValueError is raised
    with pytest.raises(ValueError, match="Input DataFrame missing required columns: \\['room_temp_c'\\]"):
         transform_meter_data(spark, input_df)

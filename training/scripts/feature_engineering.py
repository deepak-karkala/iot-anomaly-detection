'''
Script 2: Feature Engineering (scripts/feature_engineering.py)
	- Purpose: Runs as a SageMaker Processing Job (using PySpark). Reads processed data, calculates ML features, and ingests them into SageMaker Feature Store.
	- Assumptions: Processed data exists in Parquet format. Feature Store Feature Group is pre-defined (usually via Terraform or SDK). Weather/HDD data needs joining (requires input path).
'''

import argparse
import logging
import sys
import time

import pandas as pd
import sagemaker
from pyspark.sql import SparkSession
from pyspark.sql.functions import (avg, col,  # Add more functions as needed
                                   expr, lag, lit, stddev, to_date)
from pyspark.sql.window import Window
from sagemaker.feature_store.feature_group import FeatureGroup

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --------------------

def calculate_features(spark, meter_data_path, weather_data_path, start_date, end_date):
    """Reads data, calculates features, and returns a Spark DataFrame."""
    logger.info(f"Calculating features for dates {start_date} to {end_date}")
    logger.info(f"Reading meter data from: {meter_data_path}")
    logger.info(f"Reading weather data from: {weather_data_path}")

    try:
        # Read meter data, filtering by date partitions on read
        # This is more efficient than loading all and then filtering
        meter_df = spark.read.format("parquet") \
            .option("path", meter_data_path) \
            .load() \
            .where(col("event_ts").between(start_date, end_date)) # Filter early

        # Read weather/HDD data (assuming daily granularity and columns like 'date', 'hdd', 'avg_temp_c')
        # Needs partitioning by date or a date column for joining
        weather_df = spark.read.format("parquet") \
             .option("path", weather_data_path) \
             .load() \
             .withColumn("weather_date", to_date(col("date_col_name"), "yyyy-MM-dd")) # Adjust date column/format

        logger.info(f"Read {meter_df.count()} meter records and {weather_df.count()} weather records.")
        if meter_df.count() == 0:
            logger.warning("No meter data found for the specified date range.")
            return None # Or return empty DataFrame with correct schema

        # --- Feature Calculations (Examples) ---

        # 1. Aggregate meter data to daily level per apartment/building
        #    Calculating avg temp diff, total energy, total water
        daily_agg_df = meter_df.withColumn("event_date", to_date(col("event_ts"))) \
            .groupBy("building_id", "apartment_id", "event_date") \
            .agg(
                avg(col("room_temp_c") - col("setpoint_temp_c")).alias("avg_temp_diff"),
                expr("max(energy_kwh) - min(energy_kwh)").alias("daily_energy_kwh"), # Assuming cumulative energy
                expr("max(hot_water_litres) - min(hot_water_litres)").alias("daily_water_l"), # Assuming cumulative water
                # Add other aggregations: avg temp, avg setpoint etc.
            )

        # 2. Join with Weather/HDD Data
        # Ensure weather_df has unique date per building or is generic
        # Assuming weather_df date column is named 'weather_date' and has 'hdd', 'avg_temp_c'
        # May need building_id join key if weather is building-specific
        daily_df_with_weather = daily_agg_df.join(
            weather_df,
            daily_agg_df["event_date"] == weather_df["weather_date"], # Adjust join keys if needed
            "left" # Use left join to keep all meter data days
        ).select(
            daily_agg_df["*"], # Select all columns from daily meter aggregates
            weather_df["hdd"],
            weather_df["avg_temp_c"] # Select desired weather columns
        )

        # 3. Time Series Features (Lag, Rolling Avg) - Requires Window Functions
        # Define window spec (per apartment, ordered by date)
        window_spec = Window.partitionBy("building_id", "apartment_id").orderBy("event_date")

        final_features_df = daily_df_with_weather \
            .withColumn("energy_lag_1d", lag("daily_energy_kwh", 1).over(window_spec)) \
            .withColumn("energy_roll_avg_7d", avg("daily_energy_kwh").over(window_spec.rowsBetween(-6, 0))) \
            .withColumn("temp_diff_lag_1d", lag("avg_temp_diff", 1).over(window_spec))
            # Add more complex features: day of week, month, interaction terms etc.

        # 4. Add Event Time required by Feature Store (often the time feature calculation happened)
        # Use current time or the date being processed
        final_features_df = final_features_df.withColumn("feature_gen_ts", expr("current_timestamp()"))

        # 5. Select and rename columns to match Feature Store definition
        # *CRITICAL*: Column names and types must match Feature Group definition
        #             The record identifier and event time columns are mandatory.
        final_features_df = final_features_df.select(
             col("apartment_id").alias("apartment_record_id"), # Example Record Identifier
             col("feature_gen_ts").alias("event_time"),        # Feature Store Event Time
             col("event_date"),
             col("building_id"),
             col("daily_energy_kwh"),
             col("avg_temp_diff"),
             col("daily_water_l"),
             col("hdd"),
             col("avg_temp_c"),
             col("energy_lag_1d"),
             col("energy_roll_avg_7d"),
             col("temp_diff_lag_1d")
             # Select other calculated features...
        ).dropna(subset=["apartment_record_id", "event_time"]) # Must not have nulls in mandatory FS columns


        logger.info("Feature calculation complete.")
        logger.info("Final Feature Schema:")
        final_features_df.printSchema()
        final_features_df.show(5)

        return final_features_df

    except Exception as e:
        logger.error(f"Error during feature calculation: {e}", exc_info=True)
        raise

def ingest_features_to_store(feature_group_name, features_df):
    """Ingests a Spark DataFrame into SageMaker Feature Store."""
    try:
        # Convert Spark DF to Pandas DF for ingestion (Feature Store limitation)
        # This can be memory intensive for very large dataframes.
        # Consider processing in batches if necessary.
        logger.info(f"Converting Spark DataFrame to Pandas for ingestion...")
        features_pandas_df = features_df.toPandas()
        logger.info(f"Converted to Pandas DataFrame with shape: {features_pandas_df.shape}")

        if features_pandas_df.empty:
            logger.warning("Pandas DataFrame is empty, skipping ingestion.")
            return

        # Initialize SageMaker session and Feature Group object
        sagemaker_session = sagemaker.Session()
        feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=sagemaker_session)

        logger.info(f"Starting ingestion into Feature Group: {feature_group_name}")
        ingestion_manager = feature_group.ingest(
            data_frame=features_pandas_df, max_workers=4, wait=False # Increase workers for parallelism
        )
        # Wait for ingestion to complete (can take time)
        logger.info("Waiting for Feature Store ingestion to complete...")
        ingestion_manager.wait()
        failed_records = ingestion_manager.failed_rows()
        if failed_records:
             logger.error(f"Feature Store ingestion completed with {len(failed_records)} failed records.")
             # Consider logging/saving failed records for investigation
             # raise RuntimeError(f"{len(failed_records)} records failed ingestion.") # Option to fail job
        else:
             logger.info("Feature Store ingestion completed successfully.")

    except Exception as e:
        logger.error(f"Error during Feature Store ingestion: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meter-data-path", type=str, required=True)
    parser.add_argument("--weather-data-path", type=str, required=True) # Add input for weather
    parser.add_argument("--start-date", type=str, required=True, help="Format YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, required=True, help="Format YYYY-MM-DD")
    parser.add_argument("--feature-group-name", type=str, required=True)

    args = parser.parse_args()

    spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()

    try:
        features_spark_df = calculate_features(
            spark,
            args.meter_data_path,
            args.weather_data_path, # Pass weather path
            args.start_date,
            args.end_date
        )

        if features_spark_df and features_spark_df.count() > 0:
            ingest_features_to_store(args.feature_group_name, features_spark_df)
            logger.info("Feature engineering and ingestion process completed.")
        else:
            logger.warning("No features generated or DataFrame was empty. Skipping ingestion.")

    except Exception as e:
        logger.error(f"Unhandled exception during feature engineering: {e}", exc_info=True)
        sys.exit(1)
    finally:
        spark.stop()

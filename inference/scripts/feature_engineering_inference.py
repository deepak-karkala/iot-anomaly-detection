'''
Purpose: Runs as a SageMaker Processing Job daily. Reads the latest processed data
    (e.g., for the last N days needed for lags, or just the current day if features allow),
    calculates features identical to the training pipeline's feature engineering step,
    and outputs them to S3 in a format suitable for Batch Transform. Crucially, it should
    use the same logic and potentially shared code/parameters as the training feature engineering.
Difference from Training Feature Eng: This script focuses on generating features for new,
    unseen data for the specific day(s) inference is needed, rather than a historical range.
    It might read feature definitions or scaling parameters saved during training if not
    using Feature Store directly.
'''

import argparse
import logging
import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import (avg, col,  # Reuse needed functions
                                   current_timestamp, expr, lag, lit, stddev,
                                   to_date)
from pyspark.sql.window import Window

# We might need joblib to load the scaler if not using Feature Store transforms
# import joblib

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --------------------

# Base path for processing job I/O in SageMaker
BASE_PATH = "/opt/ml/processing"
PROCESSED_METER_INPUT_PATH = os.path.join(BASE_PATH, "input", "processed_meter")
PROCESSED_WEATHER_INPUT_PATH = os.path.join(BASE_PATH, "input", "processed_weather")
FEATURE_OUTPUT_PATH = os.path.join(BASE_PATH, "output", "inference_features")


def calculate_inference_features(spark, meter_data_path, weather_data_path, inference_date_str):
    """
    Calculates features for a specific date for batch inference.
    Uses logic consistent with training feature engineering.
    """
    logger.info(f"Calculating inference features for date: {inference_date_str}")
    inference_date = datetime.strptime(inference_date_str, '%Y-%m-%d').date()
    # Determine necessary lookback period for lag features (e.g., 7 days for rolling avg)
    lookback_days = 7 # Example
    start_date_dt = inference_date - timedelta(days=lookback_days)
    start_date_str = start_date_dt.strftime('%Y-%m-%d')

    logger.info(f"Reading meter data from {meter_data_path} between {start_date_str} and {inference_date_str}")
    logger.info(f"Reading weather data from {weather_data_path}") # Assume daily weather

    try:
        # Read meter data for the required range (inference date + lookback)
        meter_df = spark.read.format("parquet") \
            .option("path", meter_data_path) \
            .load() \
            .where(col("event_ts").between(start_date_str, inference_date_str + " 23:59:59")) # Filter date range

        # Read weather data (might need filtering if not daily)
        weather_df = spark.read.format("parquet") \
             .option("path", weather_data_path) \
             .load() \
             .withColumn("weather_date", to_date(col("date_col_name"), "yyyy-MM-dd")) # Adjust date column/format

        logger.info(f"Read {meter_df.count()} meter records and {weather_df.count()} weather records for the period.")
        if meter_df.where(to_date(col("event_ts")) == inference_date_str).count() == 0:
             logger.warning(f"No meter data found for the target inference date: {inference_date_str}. Exiting.")
             return None

        # --- Feature Calculations (MUST mirror training feature engineering) ---

        # 1. Aggregate meter data daily
        daily_agg_df = meter_df.withColumn("event_date", to_date(col("event_ts"))) \
            .groupBy("building_id", "apartment_id", "event_date") \
            .agg(
                avg(col("room_temp_c") - col("setpoint_temp_c")).alias("avg_temp_diff"),
                expr("max(energy_kwh) - min(energy_kwh)").alias("daily_energy_kwh"),
                expr("max(hot_water_litres) - min(hot_water_litres)").alias("daily_water_l"),
            )

        # 2. Join with Weather/HDD Data
        daily_df_with_weather = daily_agg_df.join(
            weather_df,
            daily_agg_df["event_date"] == weather_df["weather_date"],
            "left"
        ).select(
            daily_agg_df["*"],
            weather_df["hdd"],
            weather_df["avg_temp_c"]
        )

        # 3. Time Series Features (Lag, Rolling Avg)
        window_spec = Window.partitionBy("building_id", "apartment_id").orderBy("event_date")
        features_with_lags = daily_df_with_weather \
            .withColumn("energy_lag_1d", lag("daily_energy_kwh", 1).over(window_spec)) \
            .withColumn("energy_roll_avg_7d", avg("daily_energy_kwh").over(window_spec.rowsBetween(-6, 0))) \
            .withColumn("temp_diff_lag_1d", lag("avg_temp_diff", 1).over(window_spec))
            # Add ALL features calculated during training

        # --- **** IMPORTANT **** ---
        # The absolute most critical part is ensuring feature_engineering_inference.py produces features
        #    exactly as expected by the model trained using feature_engineering.py. Using SageMaker Feature Store
        #   is the best way to guarantee this. If not using Feature Store, consider refactoring the core feature
        #   logic into a shared Python library imported by both scripts.
        # If NOT using Feature Store transformations, you might need to load and apply
        # the SAME scaler fitted during training here.
        # scaler = joblib.load(scaler_path_from_model_artifact_or_s3)
        # scaled_feature_data = scaler.transform(features_df[numeric_cols])
        # This adds complexity and potential skew risk if scaler isn't handled carefully.
        # --- ******************* ---

        # 4. Filter for ONLY the target inference date
        inference_features_df = features_with_lags.where(col("event_date") == inference_date_str)

        # 5. Select ONLY the features needed by the model for inference
        #    The order might matter depending on the inference script.
        #    Also include identifiers needed later (apartment_id, building_id, event_date).
        final_cols_for_inference = [
             "apartment_id", # Keep identifiers
             "building_id",
             "event_date",
             # --- Feature columns in EXACT order model expects ---
             "daily_energy_kwh",
             "avg_temp_diff",
             "hdd",
             "avg_temp_c",
             "energy_lag_1d",
             "energy_roll_avg_7d",
             "temp_diff_lag_1d"
             # --- Ensure ALL features are selected ---
        ]
        # Handle potential missing columns if filtering reduced data significantly
        existing_cols = [c for c in final_cols_for_inference if c in inference_features_df.columns]
        inference_output_df = inference_features_df.select(*existing_cols).fillna(0) # Fill NaNs resulting from lags/joins

        logger.info(f"Feature calculation complete for inference date {inference_date_str}.")
        logger.info(f"Output Feature Schema:")
        inference_output_df.printSchema()
        inference_output_df.show(5)

        return inference_output_df

    except Exception as e:
        logger.error(f"Error during inference feature calculation: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Expecting paths mounted by SageMaker Processing Job
    parser.add_argument("--inference-date", type=str, required=True, help="Target date YYYY-MM-DD")
    # Add args for input paths if not using fixed env vars
    # parser.add_argument("--meter-data-path", type=str, default=PROCESSED_METER_INPUT_PATH)
    # parser.add_argument("--weather-data-path", type=str, default=PROCESSED_WEATHER_INPUT_PATH)

    args = parser.parse_args()
    spark = SparkSession.builder.appName("InferenceFeatureEngineering").getOrCreate()

    try:
        # Assuming input data is mounted correctly by the Processing Job definition
        inference_features_df = calculate_inference_features(
            spark,
            PROCESSED_METER_INPUT_PATH, # Reads from /opt/ml/processing/input/processed_meter
            PROCESSED_WEATHER_INPUT_PATH, # Reads from /opt/ml/processing/input/processed_weather
            args.inference_date
        )

        if inference_features_df and inference_features_df.count() > 0:
            logger.info(f"Writing {inference_features_df.count()} inference features to {FEATURE_OUTPUT_PATH}")
            # Write features to the output path mounted by SageMaker Processing
            # Batch Transform prefers CSV or JSON Lines usually, but check model container needs
            # Ensure header=False if container doesn't expect it, order must be precise
            inference_features_df.write.mode("overwrite").format("csv").option("header", "false").save(FEATURE_OUTPUT_PATH)
            logger.info("Inference features written successfully.")
        else:
            logger.warning("No inference features generated for the specified date.")

    except Exception as e:
        logger.error(f"Unhandled exception during inference feature engineering: {e}", exc_info=True)
        sys.exit(1)
    finally:
        spark.stop()

'''
- Takes source/destination paths, DB/Table names as arguments.
- Initializes Glue/Spark contexts.
- Reads JSON data from the raw S3 path using create_dynamic_frame.from_options.
    Job Bookmarks are implicitly handled via transformation_ctx.
- Converts to a Spark DataFrame for easier transformations (selecting, casting
    timestamp, adding partition columns, basic cleaning).
- Converts back to a Glue DynamicFrame.
- Uses glueContext.getSink with connection_type="s3" and enableUpdateCatalog=True
    to write partitioned Parquet data to the processed S3 path and update the
    Glue Catalog table defined in Terraform.
- Commits the job (important for bookmarks).
'''

# glue_ingest_meter_data_v2_refactored.py

import logging
import sys

from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame  # Import DynamicFrame
from awsglue.job import Job
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql.functions import col, dayofmonth, month, to_timestamp, year
from pyspark.sql.types import (DoubleType, IntegerType,  # Import Spark types
                               StringType, StructField, StructType,
                               TimestampType)

# --- Logger Setup ---
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# --------------------

def transform_meter_data(spark, df_raw, timestamp_format='yyyy-MM-dd HH:mm:ss'):
    """
    Applies transformations to the raw meter data DataFrame.

    :param spark: SparkSession object
    :param df_raw: Input Spark DataFrame with raw data
    :param timestamp_format: Format string for parsing input timestamps
    :return: Transformed Spark DataFrame ready for writing
    """
    logger.info("Starting meter data transformation.")

    # Ensure input is a DataFrame
    if not hasattr(df_raw, "select"):
         logger.error("Input to transform_meter_data is not a Spark DataFrame.")
         raise TypeError("Input df_raw must be a Spark DataFrame")

    # Define expected columns for selection (adjust as needed)
    required_cols = [
        "timestamp_str", "meter_id", "apartment_id", "building_id",
        "energy_kwh", "room_temp_c", "setpoint_temp_c", "hot_water_litres"
    ]

    # Check if all required columns exist
    missing_cols = [c for c in required_cols if c not in df_raw.columns]
    if missing_cols:
        logger.error(f"Input DataFrame missing required columns: {missing_cols}")
        raise ValueError(f"Input DataFrame missing required columns: {missing_cols}")

    # 1. Select necessary columns
    df_selected = df_raw.select(*required_cols) # Use * to unpack list

    # 2. Convert timestamp string to timestamp type
    df_selected = df_selected.withColumn("event_ts", to_timestamp(col("timestamp_str"), timestamp_format))

    # 3. Add partition columns (year, month, day)
    df_partitioned = df_selected.withColumn("year", year(col("event_ts"))) \
                                .withColumn("month", month(col("event_ts"))) \
                                .withColumn("day", dayofmonth(col("event_ts")))

    # 4. Basic Cleaning (Drop rows with null timestamp or meter_id after conversion)
    # Note: event_ts will be null if to_timestamp fails
    df_cleaned = df_partitioned.dropna(subset=["event_ts", "meter_id"])

    # 5. Select final columns for output (partitions last)
    final_cols = [
        "event_ts", "meter_id", "apartment_id", "building_id",
        "energy_kwh", "room_temp_c", "setpoint_temp_c", "hot_water_litres",
        "year", "month", "day"
    ]
    df_final = df_cleaned.select(*final_cols) # Use * to unpack list

    logger.info("Meter data transformation complete.")
    return df_final


# --- Main Glue Job Logic ---
if __name__ == "__main__":
    ## @params: [JOB_NAME, source_path, destination_path, database_name, table_name]
    args = getResolvedOptions(sys.argv, [
        'JOB_NAME',
        'source_path',
        'destination_path',
        'database_name',
        'table_name'
        ])

    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    job = Job(glueContext)
    job.init(args['JOB_NAME'], args)

    logger.info(f"Starting Glue job {args['JOB_NAME']}")
    logger.info(f"Reading raw meter data from: {args['source_path']}")
    logger.info(f"Writing processed data to: {args['destination_path']}")
    logger.info(f"Target database: {args['database_name']}, Target table: {args['table_name']}")

    try:
        input_dynamic_frame = glueContext.create_dynamic_frame.from_options(
            connection_type="s3",
            connection_options={"paths": [args['source_path']], "recurse": True},
            format="json",
            transformation_ctx="input_dynamic_frame"
        )
    except Exception as e:
        logger.error(f"Error reading from source S3 path {args['source_path']}: {e}")
        sys.exit(f"Job failed during data read: {e}")

    if input_dynamic_frame.count() == 0:
        logger.warning("No new data found in source path. Exiting job.")
        job.commit()
        sys.exit(0)

    logger.info(f"Read {input_dynamic_frame.count()} records from source.")
    df_raw = input_dynamic_frame.toDF()

    try:
        # Call the transformation function
        df_transformed = transform_meter_data(spark, df_raw)

        final_count = df_transformed.count()
        logger.info(f"Processed {final_count} records.")
        if final_count > 0:
          logger.info("Processed Schema:")
          df_transformed.printSchema()
          logger.info("Sample Processed Data (first 5 rows):")
          df_transformed.show(5, truncate=False)
        else:
          logger.warning("No records remained after processing and cleaning.")
          job.commit()
          sys.exit(0)

    except Exception as e:
        logger.error(f"Error during data transformation: {e}")
        sys.exit(f"Job failed during data transformation: {e}")

    # Convert back to DynamicFrame before writing
    output_dynamic_frame = DynamicFrame.fromDF(df_transformed, glueContext, "output_dynamic_frame")

    try:
        sink = glueContext.getSink(
            connection_type="s3",
            path=args['destination_path'],
            enableUpdateCatalog=True,
            updateBehavior="UPDATE_IN_DATABASE",
            partitionKeys=["year", "month", "day"],
            options={"database": args['database_name'], "tableName": args['table_name']},
            transformation_ctx="datasink"
        )
        sink.setFormat("glueparquet", compression="snappy")
        sink.setCatalogInfo(catalogDatabase=args['database_name'], catalogTableName=args['table_name'])
        sink.writeFrame(output_dynamic_frame)
        logger.info(f"Successfully wrote data to {args['destination_path']} and updated catalog.")
    except Exception as e:
        logger.error(f"Error writing data to destination {args['destination_path']} or updating catalog: {e}")
        sys.exit(f"Job failed during data write/catalog update: {e}")

    job.commit()
    logger.info(f"Glue job {args['JOB_NAME']} completed successfully.")

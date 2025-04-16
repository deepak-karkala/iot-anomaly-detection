'''
- Purpose:
	Runs as a Lambda function or Glue Python Shell Job.
	Reads raw anomaly scores from S3 (output of Batch Transform),
	applies thresholds, formats alerts, and writes them to DynamoDB/RDS.
- Assumptions:
	Scores are in CSV/JSON Lines format in S3. Alert database table exists.
	Thresholds are provided (e.g., via environment variables or config).
'''

import csv
import io
import json
import logging
import os
from decimal import Decimal  # For DynamoDB number compatibility

import boto3

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# --------------------

# --- Configuration ---
# Read from environment variables set in Lambda/Glue Job
ALERT_THRESHOLD = float(os.environ.get("ALERT_THRESHOLD", "5.0")) # Example threshold
ALERT_DB_TABLE_NAME = os.environ.get("ALERT_DB_TABLE_NAME", "hometech-ml-ad-alerts-dev")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")

# --- AWS Clients ---
s3_client = boto3.client('s3')
dynamodb_resource = boto3.resource('dynamodb', region_name=AWS_REGION)

def process_s3_object(bucket, key, alert_table):
    """Reads a single S3 object containing scores and generates alerts."""
    logger.info(f"Processing S3 object: s3://{bucket}/{key}")
    alerts_generated = 0
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        body = response['Body'].read().decode('utf-8')

        # Assuming output from inference.py was CSV: "id_col1,id_col2,id_col3,anomaly_score_combined"
        # Use io.StringIO to simulate a file for csv.reader
        csvfile = io.StringIO(body)
        reader = csv.reader(csvfile)

        # Use DynamoDB batch writer for efficiency
        with alert_table.batch_writer() as batch:
            for row in reader:
                try:
                    # --- Parse row based on inference output format ---
                    # Example: apartment_id, building_id, event_date_str, score
                    if len(row) != 4:
                         logger.warning(f"Skipping row with unexpected number of columns: {row}")
                         continue

                    apartment_id = row[0]
                    building_id = row[1]
                    event_date_str = row[2] # Assuming date string
                    score_str = row[3]
                    score = float(score_str)

                    # --- Apply Threshold ---
                    if score > ALERT_THRESHOLD:
                        alerts_generated += 1
                        # --- Format Alert Item for DynamoDB ---
                        alert_id = f"{apartment_id}#{event_date_str}" # Example Partition Key + Sort Key
                        item = {
                            'AlertID': alert_id, # PK
                            'ApartmentID': apartment_id, # Could be GSI PK
                            'EventDate': event_date_str, # SK or Attribute
                            'BuildingID': building_id,
                            'AnomalyScore': Decimal(str(score)), # Use Decimal for DynamoDB numbers
                            'Threshold': Decimal(str(ALERT_THRESHOLD)),
                            'AlertTimestamp': datetime.utcnow().isoformat(),
                            'Status': 'Unseen' # Initial status
                            # Add reference to model version used if available
                        }
                        logger.debug(f"Putting item: {item}")
                        batch.put_item(Item=item)

                except (ValueError, IndexError) as parse_err:
                    logger.warning(f"Skipping row due to parsing error: {row} - {parse_err}")
                    continue

        logger.info(f"Processed {reader.line_num} rows. Generated {alerts_generated} alerts from s3://{bucket}/{key}.")
        return alerts_generated

    except Exception as e:
        logger.error(f"Error processing file s3://{bucket}/{key}: {e}", exc_info=True)
        raise # Propagate error to Glue Job/Lambda caller

def lambda_handler(event, context):
    """
    Lambda handler triggered by S3 event when Batch Transform output is written.
    Can also be adapted for a Glue Python Shell job triggered differently.
    """
    logger.info(f"Received event: {json.dumps(event)}")
    alert_table = dynamodb_resource.Table(ALERT_DB_TABLE_NAME)
    total_alerts = 0
    processed_files = 0

    # Process all records in the event (handles multiple file uploads)
    for record in event.get('Records', []):
        if 's3' not in record:
            logger.warning("Skipping record without S3 information.")
            continue

        s3_info = record['s3']
        bucket = s3_info['bucket']['name']
        key = s3_info['object']['key']

        # Avoid processing non-output files if prefix is broad
        if not key.endswith(('.csv', '.out', '.jsonl')): # Adjust extensions based on Batch output
             logger.info(f"Skipping non-output file: {key}")
             continue

        try:
            alerts_generated = process_s3_object(bucket, key, alert_table)
            total_alerts += alerts_generated
            processed_files += 1
        except Exception:
            # Error logged within process_s3_object, continue to next file if possible
            logger.error(f"Failure processing {key}, continuing...") # Consider error strategy
            # Depending on requirements, might want to fail the whole invocation

    logger.info(f"Lambda execution finished. Processed {processed_files} files, generated {total_alerts} total alerts.")
    return {
        'statusCode': 200,
        'body': json.dumps(f'Processed {processed_files} files, generated {total_alerts} alerts.')
    }

# --- If running as a Glue Python Shell Job ---
# You would need getResolvedOptions for S3 input path(s) and trigger logic
# instead of the lambda_handler event structure. The core processing logic
# in process_s3_object remains similar. Example Glue Job structure:
#
# if __name__ == "__main__":
#     args = getResolvedOptions(sys.argv, ['JOB_NAME', 'input_s3_path', 'alert_table_name', 'alert_threshold'])
#     ALERT_DB_TABLE_NAME = args['alert_table_name']
#     ALERT_THRESHOLD = float(args['alert_threshold'])
#     input_path = args['input_s3_path'] # e.g., s3://bucket/path/to/batch/output/
#
#     alert_table = dynamodb_resource.Table(ALERT_DB_TABLE_NAME)
#     # List objects in the input_path prefix
#     # Paginate through results for large number of files
#     paginator = s3_client.get_paginator('list_objects_v2')
#     bucket, prefix = input_path.replace("s3://", "").split("/", 1)
#     pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
#
#     total_alerts = 0
#     processed_files = 0
#     for page in pages:
#         for obj in page.get('Contents', []):
#             key = obj['Key']
#             if key.endswith('/'): continue # Skip folders
#             # Add file extension filtering if needed
#             try:
#                 alerts = process_s3_object(bucket, key, alert_table)
#                 total_alerts += alerts
#                 processed_files += 1
#             except Exception:
#                 logger.error(f"Failure processing {key}, continuing...")
#     logger.info(f"Glue job finished. Processed {processed_files} files, generated {total_alerts} total alerts.")

import csv
import io
import json
import logging
import os
import urllib.parse  # To parse S3 path
from datetime import datetime
from decimal import Decimal

import boto3

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# --------------------

# --- Configuration ---
ALERT_THRESHOLD = float(os.environ.get("ALERT_THRESHOLD", "5.0"))
ALERT_DB_TABLE_NAME = os.environ.get("ALERT_DB_TABLE_NAME", "hometech-ml-ad-alerts-dev")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")

# --- AWS Clients ---
s3_client = boto3.client('s3', region_name=AWS_REGION)
dynamodb_resource = boto3.resource('dynamodb', region_name=AWS_REGION)

def process_s3_object(bucket, key, alert_table):
    """Reads a single S3 object containing scores and generates alerts."""
    logger.info(f"Processing S3 object: s3://{bucket}/{key}")
    alerts_generated = 0
    rows_processed = 0
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        body = response['Body'].read().decode('utf-8')

        csvfile = io.StringIO(body)
        reader = csv.reader(csvfile)

        with alert_table.batch_writer() as batch:
            for row_num, row in enumerate(reader):
                rows_processed += 1
                try:
                    if len(row) != 4: # Check expected column count
                         logger.warning(f"Skipping row {row_num+1} in {key} with unexpected columns: {row}")
                         continue

                    apartment_id = row[0]
                    building_id = row[1]
                    event_date_str = row[2]
                    score_str = row[3]
                    score = float(score_str)

                    if score > ALERT_THRESHOLD:
                        alerts_generated += 1
                        alert_id = f"{apartment_id}#{event_date_str}"
                        item = {
                            'AlertID': alert_id,
                            'ApartmentID': apartment_id,
                            'EventDate': event_date_str,
                            'BuildingID': building_id,
                            'AnomalyScore': Decimal(str(score)),
                            'Threshold': Decimal(str(ALERT_THRESHOLD)),
                            'AlertTimestamp': datetime.utcnow().isoformat(),
                            'Status': 'Unseen',
                            'S3SourceFile': f"s3://{bucket}/{key}" # Add source file info
                        }
                        logger.debug(f"Putting item: {item}")
                        batch.put_item(Item=item)

                except (ValueError, IndexError, TypeError) as parse_err:
                    logger.warning(f"Skipping row {row_num+1} in {key} due to parsing error: {row} - {parse_err}")
                    continue

        logger.info(f"Processed {rows_processed} rows. Generated {alerts_generated} alerts from s3://{bucket}/{key}.")
        return alerts_generated

    except Exception as e:
        logger.error(f"Error processing file s3://{bucket}/{key}: {e}", exc_info=True)
        raise # Propagate error

def lambda_handler(event, context):
    """
    Lambda handler triggered by Step Functions after Batch Transform.
    Processes all output files in the specified S3 path.

    Expected event:
    {
        "S3OutputPath": "s3://your-bucket/inference-output/prefix/execution-name/scores/"
        # Add other context if needed
    }
    """
    logger.info(f"Received event: {json.dumps(event)}")

    try:
        s3_output_path = event['S3OutputPath']
        if not s3_output_path or not s3_output_path.startswith("s3://"):
             raise ValueError("Missing or invalid 'S3OutputPath' in input event.")

        # Parse bucket and prefix from the S3 path
        parsed_url = urllib.parse.urlparse(s3_output_path)
        bucket = parsed_url.netloc
        # Add trailing slash if missing, remove leading slash if present
        prefix = parsed_url.path.lstrip('/').rstrip('/') + '/'
        logger.info(f"Processing results from Bucket: {bucket}, Prefix: {prefix}")

    except KeyError as e:
        logger.error(f"Missing required key in input event: {e}")
        raise ValueError(f"Input event missing required key: {e}")
    except Exception as e:
         logger.error(f"Error parsing input event: {e}")
         raise ValueError("Invalid input event format")


    alert_table = dynamodb_resource.Table(ALERT_DB_TABLE_NAME)
    total_alerts = 0
    processed_files = 0

    try:
        # List objects within the specified S3 prefix
        paginator = s3_client.get_paginator('list_objects')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                # Skip directories and potentially handle specific output file names/extensions
                if key.endswith('/') or not key.endswith(('.csv', '.out', '.csv.out')): # Adjust based on Batch Transform output naming
                    logger.debug(f"Skipping non-output object: {key}")
                    continue

                try:
                    alerts_generated = process_s3_object(bucket, key, alert_table)
                    total_alerts += alerts_generated
                    processed_files += 1
                except Exception as file_e:
                    # Log error for the specific file but continue processing others
                    logger.error(f"Failure processing file {key}, continuing... Error: {file_e}")
                    # Consider a mechanism to track files that failed processing

        logger.info(f"Lambda execution finished. Processed {processed_files} files, generated {total_alerts} total alerts from prefix {prefix}.")
        return {
            'statusCode': 200,
            'body': json.dumps({
                'processed_files': processed_files,
                'total_alerts_generated': total_alerts,
                'processed_prefix': prefix
            })
        }

    except Exception as e:
        logger.error(f"Unhandled error during S3 listing or processing loop: {e}", exc_info=True)
        raise RuntimeError("Failed to process all inference results") from e

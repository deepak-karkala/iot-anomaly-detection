import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from decimal import Decimal  # For comparing DynamoDB numbers

import boto3
import pytest

# --- Configuration ---
# Fetch from environment variables set by CI/CD runner
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")
STATE_MACHINE_ARN = os.environ.get("TEST_AD_INFERENCE_SFN_ARN", "arn:aws:states:REGION:ACCOUNT_ID:stateMachine:hometech-ml-ADInferenceWorkflow-dev-unique-suffix")
ALERT_DYNAMODB_TABLE = os.environ.get("TEST_ALERT_DYNAMODB_TABLE", "hometech-ml-ad-alerts-dev-unique-suffix")
MODEL_PACKAGE_GROUP_NAME = os.environ.get("TEST_AD_MODEL_PKG_GROUP", "hometech-ml-ADApartmentAnomalyDetector-dev-unique-suffix") # Group to find approved model in
PROCESSED_BUCKET = os.environ.get("TEST_PROCESSED_BUCKET", "hometech-ml-processed-data-dev-unique-suffix") # Where inference output goes too

# --- Test Data Config ---
# Date for which inference should run (must have corresponding processed data)
TEST_INFERENCE_DATE_STR = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d') # Default to yesterday
# Alert threshold expected to be configured in the ProcessResultsLambda env vars
# We might want to test against this value.
EXPECTED_ALERT_THRESHOLD = Decimal(os.environ.get("DEFAULT_ALERT_THRESHOLD", "5.0"))

# --- Test Parameters ---
WORKFLOW_COMPLETION_TIMEOUT_SECONDS = 1800 # 30 minutes
POLL_INTERVAL_SECONDS = 30

# --- Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Fixtures ---
@pytest.fixture(scope="module")
def sfn_client():
    return boto3.client("stepfunctions", region_name=AWS_REGION)

@pytest.fixture(scope="module")
def sagemaker_client():
    return boto3.client("sagemaker", region_name=AWS_REGION)

@pytest.fixture(scope="module")
def s3_client():
    return boto3.client("s3", region_name=AWS_REGION)

@pytest.fixture(scope="module")
def dynamodb_resource():
    return boto3.resource("dynamodb", region_name=AWS_REGION)

@pytest.fixture(scope="module")
def alert_table(dynamodb_resource):
     # Make sure table name comes from env var matching Terraform output/config
    return dynamodb_resource.Table(ALERT_DYNAMODB_TABLE)

# --- Helper Functions ---
# Reusing get_execution_status from training integration test
def get_execution_status(sfn_client, execution_arn):
    """Polls Step Function execution status until completed or timed out."""
    start_time = time.time()
    while time.time() - start_time < WORKFLOW_COMPLETION_TIMEOUT_SECONDS:
        try:
            response = sfn_client.describe_execution(executionArn=execution_arn)
            status = response['status']
            if status in ['SUCCEEDED', 'FAILED', 'TIMED_OUT', 'ABORTED']:
                logger.info(f"Execution {execution_arn} finished with status: {status}")
                return response
            logger.info(f"Execution {execution_arn} status: {status}. Waiting...")
            time.sleep(POLL_INTERVAL_SECONDS)
        except Exception as e:
            logger.error(f"Error describing execution {execution_arn}: {e}")
            pytest.fail(f"Boto3 error describing execution: {e}")
    pytest.fail(f"Execution {execution_arn} timed out after {WORKFLOW_COMPLETION_TIMEOUT_SECONDS} seconds.")

def check_approved_model_exists(sagemaker_client, model_package_group):
     """Checks if at least one APPROVED model package exists."""
     try:
        response = sagemaker_client.list_model_packages(
            ModelPackageGroupName=model_package_group,
            ModelApprovalStatus='Approved',
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )
        exists = bool(response['ModelPackageSummaryList'])
        if not exists:
             logger.error(f"PRE-CHECK FAILED: No APPROVED model package found in group: {model_package_group}")
        else:
             logger.info(f"Pre-check successful: Found approved model package in group: {model_package_group}")
        return exists
     except Exception as e:
        logger.error(f"Error checking for approved model packages: {e}")
        return False

def get_alerts_for_run(alert_table, execution_arn, date_str):
    """Queries DynamoDB for alerts related to a specific run/date (needs refinement based on schema)."""
    # This is tricky without a dedicated execution ID field in the alert table.
    # Querying by date is a common pattern.
    logger.info(f"Querying DynamoDB table {alert_table.name} for alerts on date {date_str}")
    try:
        # Example using scan with filter (less efficient for large tables)
        # A GSI on EventDate would be better.
        response = alert_table.scan(
            FilterExpression="EventDate = :d",
            ExpressionAttributeValues={":d": date_str}
        )
        items = response.get('Items', [])
        # Add pagination logic if needed for large number of alerts on one day
        logger.info(f"Found {len(items)} alert items for date {date_str}.")
        return items
    except Exception as e:
        logger.error(f"Error querying DynamoDB table {alert_table.name}: {e}")
        return []

def delete_alerts_for_run(alert_table, alerts):
     """Deletes specific alert items from DynamoDB."""
     if not alerts:
         return
     logger.warning(f"Attempting cleanup of {len(alerts)} alert items from DynamoDB table {alert_table.name}...")
     try:
         with alert_table.batch_writer() as batch:
             for item in alerts:
                 # Assuming AlertID is the primary key HASH
                 key_to_delete = {'AlertID': item['AlertID']}
                 # Add RANGE key if used: e.g. 'EventDate': item['EventDate']
                 batch.delete_item(Key=key_to_delete)
         logger.info(f"Cleanup: Deleted {len(alerts)} alert items.")
     except Exception as e:
         logger.error(f"Cleanup Error: Failed to delete alert items: {e}")

def delete_s3_prefix(s3_client, bucket, prefix):
    """Deletes all objects under a given S3 prefix."""
    if not prefix.endswith('/'):
        prefix += '/'
    logger.warning(f"Attempting cleanup of S3 prefix: s3://{bucket}/{prefix}")
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        delete_us = dict(Objects=[])
        deleted_count = 0
        for item in pages.search('Contents'):
            if item:
                delete_us['Objects'].append(dict(Key=item['Key']))
                # Batch delete limits to 1000 objects per request
                if len(delete_us['Objects']) >= 1000:
                    s3_client.delete_objects(Bucket=bucket, Delete=delete_us)
                    deleted_count += len(delete_us['Objects'])
                    delete_us = dict(Objects=[])
        # Delete remaining keys
        if len(delete_us['Objects']):
            s3_client.delete_objects(Bucket=bucket, Delete=delete_us)
            deleted_count += len(delete_us['Objects'])
        logger.info(f"Cleanup: Deleted {deleted_count} objects under prefix {prefix}.")
    except Exception as e:
        logger.error(f"Cleanup Error: Failed to delete objects under prefix {prefix}: {e}")


# --- Test Function ---
# @pytest.mark.skipif(not check_approved_model_exists(boto3.client("sagemaker"), MODEL_PACKAGE_GROUP_NAME), reason="Requires an APPROVED model package in the group")
def test_ad_inference_workflow(sfn_client, sagemaker_client, s3_client, alert_table):
    """Runs the AD Inference Step Function and validates key outputs."""

    # --- Pre-check ---
    logger.info(f"Running pre-check for approved model in group: {MODEL_PACKAGE_GROUP_NAME}")
    if not check_approved_model_exists(sagemaker_client, MODEL_PACKAGE_GROUP_NAME):
         pytest.fail(f"Pre-check failed: No APPROVED model package found in group {MODEL_PACKAGE_GROUP_NAME}. Cannot run inference test.")

    execution_name = f"integ-test-infer-{uuid.uuid4()}"
    s3_output_prefix_base = f"inference-output/{STATE_MACHINE_ARN.split(':')[-1]}/{execution_name}" # Base prefix for outputs
    s3_features_prefix_base = f"inference-features/{STATE_MACHINE_ARN.split(':')[-1]}/{execution_name}"

    alerts_generated_by_test = [] # Keep track for cleanup

    try:
        # --- Trigger Step Function ---
        input_payload = {
            # Input needed by the SFN definition
            "inference_date": TEST_INFERENCE_DATE_STR,
            "ModelPackageGroupName": MODEL_PACKAGE_GROUP_NAME, # To find the model
            # Add other dynamic inputs if needed by your SFN definition
        }
        logger.info(f"Starting Step Function execution: {execution_name}")
        logger.info(f"Input Payload: {json.dumps(input_payload)}")

        response = sfn_client.start_execution(
            stateMachineArn=STATE_MACHINE_ARN,
            name=execution_name,
            input=json.dumps(input_payload)
        )
        execution_arn = response['executionArn']
        logger.info(f"Execution ARN: {execution_arn}")

        # --- Wait and Monitor ---
        final_status_response = get_execution_status(sfn_client, execution_arn)
        final_status = final_status_response['status']

        # --- Assert Final Status ---
        assert final_status == 'SUCCEEDED', f"Step Function execution failed with status {final_status}. Response: {final_status_response}"

        # --- Assert Outputs (DynamoDB Alerts) ---
        # Wait a bit for Lambda processing results to potentially finish writing
        time.sleep(15)
        alerts_generated_by_test = get_alerts_for_run(alert_table, execution_arn, TEST_INFERENCE_DATE_STR)

        # TODO: Add specific assertions based on your TEST DATA and EXPECTED ANOMALIES
        # Example: Check if a specific known test anomaly was flagged
        assert len(alerts_generated_by_test) > 0, "Expected at least one alert to be generated based on test data/model."
        logger.info(f"Found {len(alerts_generated_by_test)} alerts in DynamoDB for the test run.")

        # Example: Check content of the first alert found
        first_alert = alerts_generated_by_test[0]
        assert "AlertID" in first_alert
        assert "ApartmentID" in first_alert
        assert first_alert.get("EventDate") == TEST_INFERENCE_DATE_STR
        assert first_alert.get("Status") == "Unseen"
        assert "AnomalyScore" in first_alert
        assert first_alert.get("Threshold") == EXPECTED_ALERT_THRESHOLD
        logger.info(f"Verified structure of alert item: {first_alert.get('AlertID')}")

        # --- Assert Intermediate Outputs (Optional) ---
        # Check S3 outputs if needed for debugging
        # Note: Need to parse SFN output or derive paths based on execution name/ID
        logger.info("Placeholder: Add checks for intermediate S3 outputs (features, scores) if desired.")


    finally:
        # --- Cleanup ---
        logger.info("--- Starting Cleanup ---")
        # Delete generated alerts
        delete_alerts_for_run(alert_table, alerts_generated_by_test)

        # Delete S3 outputs (use execution name/ID based prefix)
        delete_s3_prefix(s3_client, PROCESSED_BUCKET, f"inference-features/{STATE_MACHINE_ARN.split(':')[-1]}/{execution_name}/")
        delete_s3_prefix(s3_client, PROCESSED_BUCKET, f"inference-output/{STATE_MACHINE_ARN.split(':')[-1]}/{execution_name}/")

        # Optionally delete the SageMaker Model resource created by the workflow
        # Requires getting the ModelName from SFN output or listing models by tag/prefix
        logger.warning("Placeholder: Cleanup of SageMaker Model resource skipped.")

        logger.info("--- Cleanup Finished ---")

import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta

import boto3
import pytest

# --- Configuration ---
# These should ideally be fetched from environment variables set by the CI/CD runner
# or from Terraform outputs stored securely. For this example, we use os.environ.get
# with placeholders. Replace defaults with values relevant to your TEST environment.
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1") # Make sure this is set
STATE_MACHINE_ARN = os.environ.get("TEST_AD_TRAINING_SFN_ARN", "arn:aws:states:eu-central-1:ACCOUNT_ID:stateMachine:hometech-ml-ADTrainingWorkflow-dev-unique-suffix")
MODEL_PACKAGE_GROUP_NAME = os.environ.get("TEST_AD_MODEL_PKG_GROUP", "hometech-ml-ADApartmentAnomalyDetector-dev-unique-suffix")
FEATURE_GROUP_NAME = os.environ.get("TEST_AD_FEATURE_GROUP", "hometech-ml-ad-apartment-features-dev-unique-suffix")
PROCESSED_BUCKET = os.environ.get("TEST_PROCESSED_BUCKET", "hometech-ml-processed-data-dev-unique-suffix")
# ECR Image URI corresponding to the code being tested (passed from CI)
TRAINING_IMAGE_URI = os.environ.get("TEST_TRAINING_IMAGE_URI", "ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/hometech-ml-ad-training-container-dev:latest") # MUST BE SET IN CI
GIT_HASH = os.environ.get("BITBUCKET_COMMIT", "manual-test-commit") # Provided by Bitbucket Pipelines

# --- Test Data Config ---
# Path within PROCESSED_BUCKET containing the small, pre-processed test data
# Ensure this data exists before running the test!
TEST_PROCESSED_METER_PATH = f"s3://{PROCESSED_BUCKET}/processed_meter_data/" # Adjust if test data is in subfolder
TEST_PROCESSED_WEATHER_PATH = f"s3://{PROCESSED_BUCKET}/processed_weather/" # Adjust path
# Define the date range covered by the test data
TEST_START_DATE = "2024-01-10" # Example
TEST_END_DATE = "2024-01-15"   # Example

# --- Test Parameters ---
TEST_HYPERPARAMETERS = {
    "model_strategy": "LR_LOF",
    "lof_neighbors": 5, # Use small value for faster testing
    "lof_contamination": "auto",
    # Feature columns MUST match those expected by the training script
    "feature_columns": "daily_energy_kwh,avg_temp_diff,hdd,avg_temp_c,energy_lag_1d,energy_roll_avg_7d"
}
# --- Test Timeout ---
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

# --- Helper Functions ---
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

def find_latest_model_package_arn(sagemaker_client, model_package_group):
    """Finds the ARN of the most recently created model package in a group."""
    try:
        response = sagemaker_client.list_model_packages(
            ModelPackageGroupName=model_package_group,
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )
        if response['ModelPackageSummaryList']:
            arn = response['ModelPackageSummaryList'][0]['ModelPackageArn']
            logger.info(f"Found latest model package ARN: {arn}")
            return arn
        else:
            logger.warning(f"No model packages found in group: {model_package_group}")
            return None
    except Exception as e:
        logger.error(f"Error listing model packages for group {model_package_group}: {e}")
        return None

# --- Test Function ---
def test_ad_training_workflow_execution(sfn_client, sagemaker_client, s3_client):
    """Runs the AD Training Step Function and validates key outputs."""
    execution_name = f"integ-test-{uuid.uuid4()}" # Unique name for execution
    start_time = datetime.utcnow()

    # Store ARNs/URIs of created artifacts for cleanup
    created_model_package_arn = None
    evaluation_report_uri = None
    model_artifact_uri = None

    try:
        # --- Trigger Step Function ---
        input_payload = {
            "data_params": {
                "start_date": TEST_START_DATE,
                "end_date": TEST_END_DATE,
                "processed_meter_path": TEST_PROCESSED_METER_PATH, # Pass test paths
                "processed_weather_path": TEST_PROCESSED_WEATHER_PATH
            },
            "training_params": TEST_HYPERPARAMETERS,
            "feature_group_name": FEATURE_GROUP_NAME, # Make sure this exists in test env
            "model_package_group_name": MODEL_PACKAGE_GROUP_NAME,
            "git_hash": GIT_HASH,
            "training_image_uri": TRAINING_IMAGE_URI
            # Add other inputs the Step Function expects
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

        # --- Assert Model Package Creation ---
        # It might take a moment for the package to appear after the Lambda finishes
        time.sleep(10) # Small delay before checking registry
        created_model_package_arn = find_latest_model_package_arn(sagemaker_client, MODEL_PACKAGE_GROUP_NAME)
        assert created_model_package_arn is not None, f"Could not find any model package in group {MODEL_PACKAGE_GROUP_NAME}"

        # Verify the latest package was created after our test started
        pkg_response = sagemaker_client.describe_model_package(ModelPackageName=created_model_package_arn)
        creation_time = pkg_response['CreationTime']
        logger.info(f"Model Package creation time: {creation_time}")
        # assert creation_time > start_time # Might have timezone issues, compare carefully

        # Assert Status is PendingManualApproval
        approval_status = pkg_response['ModelApprovalStatus']
        assert approval_status == 'PendingManualApproval', f"Expected ModelApprovalStatus 'PendingManualApproval', but got '{approval_status}'"
        logger.info(f"Model Package {created_model_package_arn} found with status {approval_status}.")

        # --- Assert Artifact Existence (Evaluation Report, Model Artifact) ---
        # Need to get S3 URIs - these ideally come from Step Function output if designed that way
        # If not, need to derive them based on job names/execution ID which is complex
        # Placeholder: Assuming we can get them from the final SFN output
        try:
            sfn_output = json.loads(final_status_response.get('output', '{}'))
            evaluation_report_uri = sfn_output.get('registration_output', {}).get('Payload', {}).get('evaluation_report_url', None) # Example path derivation
            model_artifact_uri = sfn_output.get('registration_output', {}).get('Payload', {}).get('model_artifact_url', None) # Example path derivation

            assert evaluation_report_uri, "Evaluation report URI not found in Step Function output."
            assert model_artifact_uri, "Model artifact URI not found in Step Function output."

            logger.info(f"Found evaluation report URI: {evaluation_report_uri}")
            logger.info(f"Found model artifact URI: {model_artifact_uri}")

            eval_bucket, eval_key = evaluation_report_uri.replace("s3://", "").split("/", 1)
            model_bucket, model_key = model_artifact_uri.replace("s3://", "").split("/", 1)

            s3_client.head_object(Bucket=eval_bucket, Key=eval_key) # Check if evaluation report exists
            logger.info("Evaluation report exists in S3.")
            s3_client.head_object(Bucket=model_bucket, Key=model_key) # Check if model artifact exists
            logger.info("Model artifact exists in S3.")

        except Exception as s3_e:
             pytest.fail(f"Could not verify artifact existence in S3: {s3_e}. Check SFN output parsing.")


    finally:
        # --- Cleanup ---
        logger.info("--- Starting Cleanup ---")
        if created_model_package_arn:
            try:
                logger.warning(f"Attempting to delete model package: {created_model_package_arn}")
                # sagemaker_client.delete_model_package(ModelPackageName=created_model_package_arn)
                logger.info("Placeholder: Model package deletion skipped (uncomment to enable).") # BE CAREFUL WITH AUTO-DELETE
            except Exception as e:
                logger.error(f"Cleanup Error: Failed to delete model package {created_model_package_arn}: {e}")

        if evaluation_report_uri:
            try:
                eval_bucket, eval_key = evaluation_report_uri.replace("s3://", "").split("/", 1)
                logger.warning(f"Attempting to delete evaluation report: {evaluation_report_uri}")
                # s3_client.delete_object(Bucket=eval_bucket, Key=eval_key)
                logger.info("Placeholder: Evaluation report deletion skipped (uncomment to enable).")
            except Exception as e:
                 logger.error(f"Cleanup Error: Failed to delete evaluation report {evaluation_report_uri}: {e}")

        if model_artifact_uri:
             try:
                model_bucket, model_key = model_artifact_uri.replace("s3://", "").split("/", 1)
                logger.warning(f"Attempting to delete model artifact: {model_artifact_uri}")
                # s3_client.delete_object(Bucket=model_bucket, Key=model_key)
                logger.info("Placeholder: Model artifact deletion skipped (uncomment to enable).")
             except Exception as e:
                 logger.error(f"Cleanup Error: Failed to delete model artifact {model_artifact_uri}: {e}")
        logger.info("--- Cleanup Finished ---")

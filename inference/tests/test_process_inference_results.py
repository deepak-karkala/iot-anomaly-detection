'''
- Verify the alert generation logic based on scores read from a mock S3 object content.
	Test thresholding and DynamoDB item formatting. Requires mocking boto3
'''

import csv
import io
from datetime import datetime
from decimal import Decimal

import boto3
import pytest
from moto import mock_aws  # Use moto for mocking AWS services
# Assuming the script is saved as 'process_inference_results.py'
# Adjust the import path if necessary
from process_inference_results import process_s3_object  # Import function

# --- Test Constants ---
TEST_BUCKET = "test-inference-output-bucket"
TEST_TABLE_NAME = "test-hometech-alerts"
TEST_REGION = "eu-central-1"

# --- Fixtures ---
@pytest.fixture(scope="function") # Use function scope for AWS mocks
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = TEST_REGION

@pytest.fixture(scope="function")
def mocked_s3(aws_credentials):
    """Mock S3 service."""
    with mock_aws():
        s3_client = boto3.client("s3", region_name=TEST_REGION)
        s3_client.create_bucket(
            Bucket=TEST_BUCKET,
            CreateBucketConfiguration={'LocationConstraint': TEST_REGION}
        )
        yield s3_client

@pytest.fixture(scope="function")
def mocked_dynamodb(aws_credentials):
    """Mock DynamoDB service and create table."""
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name=TEST_REGION)
        dynamodb.create_table(
            TableName=TEST_TABLE_NAME,
            KeySchema=[
                {'AttributeName': 'AlertID', 'KeyType': 'HASH'}, # Example PK
                # {'AttributeName': 'EventDate', 'KeyType': 'RANGE'} # Example SK
            ],
            AttributeDefinitions=[
                {'AttributeName': 'AlertID', 'AttributeType': 'S'},
                # {'AttributeName': 'EventDate', 'AttributeType': 'S'}
            ],
            ProvisionedThroughput={'ReadCapacityUnits': 1, 'WriteCapacityUnits': 1}
        )
        yield dynamodb.Table(TEST_TABLE_NAME)

# --- Test Cases ---

def test_process_s3_object_generate_alerts(mocked_s3, mocked_dynamodb):
    """Test generating alerts for scores above threshold."""
    # Prepare mock S3 object content (CSV as defined in inference.py output_fn)
    # Format: "id_col1,id_col2,id_col3,anomaly_score_combined"
    csv_content = io.StringIO()
    writer = csv.writer(csv_content)
    writer.writerow(["apt1", "bldgA", "2024-01-18", "6.5"]) # Above threshold 5.0
    writer.writerow(["apt2", "bldgA", "2024-01-18", "3.2"]) # Below threshold 5.0
    writer.writerow(["apt3", "bldgB", "2024-01-19", "9.9"]) # Above threshold 5.0
    writer.writerow(["apt4", "bldgB", "2024-01-19", "invalid_score"]) # Invalid score
    writer.writerow(["apt5", "bldgC", "2024-01-20"]) # Wrong number of columns

    s3_key = "inference-output/results.csv"
    mocked_s3.put_object(Bucket=TEST_BUCKET, Key=s3_key, Body=csv_content.getvalue().encode('utf-8'))

    # Set environment variable for threshold
    os.environ["ALERT_THRESHOLD"] = "5.0"
    os.environ["ALERT_DB_TABLE_NAME"] = TEST_TABLE_NAME # Ensure consistency

    alerts_generated = process_s3_object(TEST_BUCKET, s3_key, mocked_dynamodb)

    # Assertions
    assert alerts_generated == 2 # Only apt1 and apt3 should trigger alerts

    # Verify DynamoDB content
    response1 = mocked_dynamodb.get_item(Key={'AlertID': 'apt1#2024-01-18'})
    assert 'Item' in response1
    assert response1['Item']['ApartmentID'] == 'apt1'
    assert response1['Item']['AnomalyScore'] == Decimal("6.5")
    assert response1['Item']['Status'] == 'Unseen'

    response2 = mocked_dynamodb.get_item(Key={'AlertID': 'apt2#2024-01-18'})
    assert 'Item' not in response2 # Should not exist

    response3 = mocked_dynamodb.get_item(Key={'AlertID': 'apt3#2024-01-19'})
    assert 'Item' in response3
    assert response3['Item']['ApartmentID'] == 'apt3'
    assert response3['Item']['AnomalyScore'] == Decimal("9.9")

def test_process_s3_object_no_alerts(mocked_s3, mocked_dynamodb):
    """Test when no scores are above the threshold."""
    csv_content = io.StringIO()
    writer = csv.writer(csv_content)
    writer.writerow(["apt1", "bldgA", "2024-01-18", "1.5"])
    writer.writerow(["apt2", "bldgA", "2024-01-18", "4.9"])

    s3_key = "inference-output/no_alerts.csv"
    mocked_s3.put_object(Bucket=TEST_BUCKET, Key=s3_key, Body=csv_content.getvalue().encode('utf-8'))

    os.environ["ALERT_THRESHOLD"] = "5.0"
    os.environ["ALERT_DB_TABLE_NAME"] = TEST_TABLE_NAME

    alerts_generated = process_s3_object(TEST_BUCKET, s3_key, mocked_dynamodb)

    # Assertions
    assert alerts_generated == 0
    scan_response = mocked_dynamodb.scan()
    assert scan_response['Count'] == 0 # Table should be empty


def test_process_s3_object_empty_file(mocked_s3, mocked_dynamodb):
    """Test processing an empty S3 file."""
    csv_content = ""
    s3_key = "inference-output/empty.csv"
    mocked_s3.put_object(Bucket=TEST_BUCKET, Key=s3_key, Body=csv_content.encode('utf-8'))

    os.environ["ALERT_THRESHOLD"] = "5.0"
    os.environ["ALERT_DB_TABLE_NAME"] = TEST_TABLE_NAME

    alerts_generated = process_s3_object(TEST_BUCKET, s3_key, mocked_dynamodb)

    # Assertions
    assert alerts_generated == 0
    scan_response = mocked_dynamodb.scan()
    assert scan_response['Count'] == 0

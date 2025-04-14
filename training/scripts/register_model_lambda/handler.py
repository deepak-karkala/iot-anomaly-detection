'''
Script 5: Register Model Lambda (scripts/register_model_lambda/handler.py)
	- Purpose:
		Runs as an AWS Lambda function. Triggered by Step Functions.
		Reads evaluation results, gathers metadata, and registers a model package in SageMaker Model Registry.
	- Assumptions:
		Lambda execution role has permissions for SageMaker (CreateModelPackage) and S3 (GetObject).
		Event payload contains necessary URIs and metadata.
'''

import json
import logging
import os
import time

import boto3

# --- Logger Setup ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# --------------------

sagemaker_client = boto3.client('sagemaker')
s3_client = boto3.client('s3')

def get_evaluation_report(s3_uri):
    """Downloads and parses the evaluation report from S3."""
    try:
        bucket, key = s3_uri.replace("s3://", "").split("/", 1)
        response = s3_client.get_object(Bucket=bucket, Key=key)
        report_str = response['Body'].read().decode('utf-8')
        report = json.loads(report_str)
        logger.info(f"Successfully downloaded and parsed evaluation report from {s3_uri}")
        return report
    except Exception as e:
        logger.error(f"Failed to get evaluation report from {s3_uri}: {e}")
        # Depending on requirements, might return None or raise error
        return {"error": f"Failed to load evaluation report: {e}"}


def lambda_handler(event, context):
    """
    Lambda handler function to register a model package.

    Expected event structure:
    {
        "model_artifact_url": "s3://...",
        "evaluation_report_url": "s3://...",
        "model_package_group_name": "...",
        "git_hash": "...",
        "feature_group_name": "...",
        "training_params": { ... }, // Hyperparameters etc.
        "data_params": { // Info about data used
             "start_date": "...",
             "end_date": "..."
        },
        "image_uri": "..." // Docker image used for training
    }
    """
    logger.info(f"Received event: {json.dumps(event)}")

    try:
        model_s3_uri = event['model_artifact_url']
        eval_report_s3_uri = event['evaluation_report_url']
        model_package_group_name = event['model_package_group_name']
        git_hash = event['git_hash']
        feature_group_name = event['feature_group_name']
        training_params = event['training_params'] # Hyperparameters
        data_params = event['data_params']
        image_uri = event['image_uri'] # Training container image

        # --- Get Evaluation Metrics ---
        evaluation_report = get_evaluation_report(eval_report_s3_uri)
        if not evaluation_report or "error" in evaluation_report:
             # Handle inability to read report - maybe register anyway but flag it?
             logger.error("Cannot proceed without valid evaluation report.")
             raise ValueError("Evaluation report could not be loaded.")

        # --- Prepare Model Package Input ---
        model_package_description = (
            f"AD Model trained on {data_params.get('start_date','N/A')} to {data_params.get('end_date','N/A')}. "
            f"Code Commit: {git_hash}. Feature Group: {feature_group_name}. "
            f"Evaluation Status: {evaluation_report.get('status', 'Unknown')}"
        )

        # Use inference specification matching how Batch Transform will use the model
        inference_spec = {
            "Containers": [
                {
                    "Image": image_uri,
                    "ModelDataUrl": model_s3_uri
                }
            ],
            "SupportedContentTypes": ["text/csv", "application/json", "application/x-parquet"], # Adjust as needed
            "SupportedResponseMIMETypes": ["text/csv", "application/json", "application/x-parquet"], # Adjust as needed
        }

        # Custom metadata properties for lineage and reproducibility
        custom_properties = {
            "GitCommit": git_hash,
            "FeatureGroupName": feature_group_name,
            "TrainingStartDate": data_params.get('start_date', 'N/A'),
            "TrainingEndDate": data_params.get('end_date', 'N/A'),
            # Add all hyperparameters and evaluation metrics
            **{f"Hyperparameter_{k}": str(v) for k, v in training_params.items()},
            **{f"Evaluation_{k}": str(v) for k, v in evaluation_report.items()}
        }

        model_package_input = {
            "ModelPackageGroupName": model_package_group_name,
            "ModelPackageDescription": model_package_description,
            "ModelApprovalStatus": "PendingManualApproval", # Set initial status
            "InferenceSpecification": inference_spec,
            "MetadataProperties": {
                # CommitID is specifically indexed by SageMaker
                "CommitId": git_hash
            },
            "CustomerMetadataProperties": custom_properties,
             # Add DriftCheckBaselines here if using Model Monitor
        }

        logger.info(f"Creating model package with input: {json.dumps(model_package_input)}")
        response = sagemaker_client.create_model_package(**model_package_input)
        model_package_arn = response['ModelPackageArn']

        logger.info(f"Successfully created Model Package: {model_package_arn}")
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Model package registered successfully',
                'modelPackageArn': model_package_arn
            })
        }

    except KeyError as e:
        logger.error(f"Missing required key in input event: {e}")
        return {'statusCode': 400, 'body': f"Missing input key: {e}"}
    except Exception as e:
        logger.error(f"Failed to register model package: {e}", exc_info=True)
        # Raise error to signal failure to Step Functions
        raise e

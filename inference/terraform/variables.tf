variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "eu-central-1"
}

variable "project_name" {
  description = "Base name for resources (e.g., hometech-ml)"
  type        = string
  default     = "hometech-ml"
}

variable "env_suffix" {
  description = "Environment suffix (e.g., dev, prod, or unique id)"
  type        = string
  default     = "dev-unique-suffix" # MUST MATCH other stacks in the same env
}

# --- Inputs from OTHER Stacks / Global Config ---
variable "processed_bucket_name" {
  description = "Name of the S3 bucket for processed data (from ingestion)"
  type        = string
  # Example: pass using -var="processed_bucket_name=hometech-ml-processed-data-dev-unique-suffix"
}

variable "scripts_bucket_name" {
  description = "Name of the S3 bucket holding Lambda/Glue scripts (from ingestion/shared)"
  type        = string
}

variable "ad_model_package_group_name" {
  description = "Name of the SageMaker Model Package Group containing approved AD models (from training)"
  type        = string
  # Example: pass using -var="ad_model_package_group_name=hometech-ml-ADApartmentAnomalyDetector-dev-unique-suffix"
}

variable "training_image_uri" {
  description = "Full URI of the Docker image used for training AND inference (must contain inference.py)"
  type        = string
   # Example: pass using -var="training_image_uri=..."
}

variable "sagemaker_processing_role_arn" {
  description = "ARN of the IAM role for SageMaker Processing Jobs (from training stack or shared)"
  type        = string
   # Example: pass using -var="sagemaker_processing_role_arn=..."
}

variable "spark_processing_image_uri" {
  description = "URI of the Spark Processing image for feature engineering"
  type        = string
  default = "YOUR_SPARK_PROCESSING_IMAGE_URI" # REPLACE THIS - Same as used in training
}


# --- Inference Specific Config ---
variable "alert_db_table_name" {
  description = "Name for the DynamoDB table storing AD alerts"
  type        = string
  default     = "ad-alerts" # Will have project/env added
}

variable "inference_sfn_name" {
  description = "Name of the Step Functions State Machine for AD Inference"
  type        = string
  default     = "ADInferenceWorkflow" # Will have project/env added
}

variable "scheduler_name" {
  description = "Name of the EventBridge Scheduler"
  type        = string
  default     = "DailyADInferenceTrigger" # Will have project/env added
}

variable "scheduler_expression" {
  description = "Cron expression for the daily trigger (UTC)"
  type        = string
  default     = "cron(0 5 * * ? *)" # Example: Run at 5:00 AM UTC daily
}

variable "batch_transform_instance_type" {
  description = "Instance type for SageMaker Batch Transform Jobs"
  type        = string
  default     = "ml.m5.large"
}

variable "batch_transform_instance_count" {
  description = "Instance count for SageMaker Batch Transform Jobs"
  type        = number
  default     = 1
}

variable "default_alert_threshold" {
  description = "Default anomaly score threshold for alerting"
  type        = string # Passed as string env var to Lambda
  default     = "5.0"
}

# --- Lambda Code Paths ---
variable "lambda_get_model_code_dir" {
  description = "Local directory for get_approved_model lambda"
  type        = string
  default     = "../../scripts/get_approved_model_lambda"
}
variable "lambda_create_sm_model_code_dir" {
  description = "Local directory for create_sagemaker_model lambda"
  type        = string
  default     = "../../scripts/create_sagemaker_model_lambda"
}
variable "lambda_process_results_code_dir" {
  description = "Local directory for process_inference_results lambda"
  type        = string
  default     = "../../scripts/process_inference_results_lambda"
}
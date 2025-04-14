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
  default     = "dev-unique-suffix" # MUST MATCH the suffix used in ingestion infra
}

# --- Inputs from Ingestion Infra ---
variable "processed_bucket_name" {
  description = "Name of the S3 bucket for processed data (from ingestion)"
  type        = string
}

variable "scripts_bucket_name" {
  description = "Name of the S3 bucket holding Glue/Lambda scripts (from ingestion)"
  type        = string
}

variable "glue_catalog_db_name" {
  description = "Name of the Glue Data Catalog database (from ingestion)"
  type        = string
}

# --- Feature Store ---
variable "ad_feature_group_name" {
  description = "Name for the SageMaker Feature Group for AD features"
  type        = string
  default     = "ad-apartment-features" # Will have env suffix added
}

# --- ECR & Training Container ---
variable "ecr_repo_name" {
  description = "Name for the ECR repository for the training container"
  type        = string
  default     = "ad-training-container" # Will have project/env added
}

variable "training_image_uri" {
  description = "Full URI of the Docker image in ECR for training (e.g., <account_id>.dkr.ecr.<region>.amazonaws.com/<repo_name>:latest)"
  type        = string
  # This needs to be provided AFTER the image is built and pushed
  # Can be passed as a tfvar or determined via data source/output if built in TF
  default = "YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/hometech-ml-ad-training-container-dev-unique-suffix:latest" # REPLACE THIS
}


# --- Model Registry ---
variable "ad_model_package_group_name" {
  description = "Name for the SageMaker Model Package Group for AD models"
  type        = string
  default     = "ADApartmentAnomalyDetector" # Will have env suffix added
}

# --- Lambda ---
variable "lambda_code_dir" {
  description = "Local directory containing the registration lambda code"
  type        = string
  default     = "../../scripts/register_model_lambda" # Relative path example
}

variable "lambda_zip_name" {
  description = "Name of the zip file for the lambda function"
  type        = string
  default     = "register_model_lambda.zip"
}

variable "lambda_function_name" {
  description = "Name of the Lambda function"
  type        = string
  default     = "RegisterADModelFunction" # Will have project/env added
}


# --- Step Functions ---
variable "ad_training_sfn_name" {
  description = "Name of the Step Functions State Machine for AD Training"
  type        = string
  default     = "ADTrainingWorkflow" # Will have project/env added
}

# --- SageMaker Job Resources ---
variable "processing_instance_type" {
  description = "Instance type for SageMaker Processing Jobs"
  type        = string
  default     = "ml.m5.large"
}

variable "processing_instance_count" {
  description = "Instance count for SageMaker Processing Jobs"
  type        = number
  default     = 1
}

variable "training_instance_type" {
  description = "Instance type for SageMaker Training Jobs"
  type        = string
  default     = "ml.m5.large"
}

variable "training_instance_count" {
  description = "Instance count for SageMaker Training Jobs"
  type        = number
  default     = 1
}
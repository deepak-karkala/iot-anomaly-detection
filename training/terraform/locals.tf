locals {
  # Construct names with project and environment context
  ad_feature_group_name_full    = lower("${var.project_name}-${var.ad_feature_group_name}-${var.env_suffix}")
  ecr_repo_name_full            = lower("${var.project_name}-${var.ecr_repo_name}-${var.env_suffix}")
  ad_model_package_group_name_full = "${var.project_name}-${var.ad_model_package_group_name}-${var.env_suffix}" # Hyphens allowed here
  lambda_function_name_full     = "${var.project_name}-${var.lambda_function_name}-${var.env_suffix}"
  lambda_zip_output_path        = "${path.module}/${var.lambda_zip_name}" # Local path for zip creation
  lambda_s3_key                 = "lambda-code/${var.lambda_zip_name}"
  ad_training_sfn_name_full     = "${var.project_name}-${var.ad_training_sfn_name}-${var.env_suffix}"

  # Construct S3 URIs used in Step Functions
  s3_processed_meter_uri = "s3://${var.processed_bucket_name}/processed_meter_data/"
  s3_offline_store_uri   = "s3://${var.processed_bucket_name}/feature-store-offline/${local.ad_feature_group_name_full}/"
  # Define where evaluation reports are stored by the processing job
  s3_evaluation_output_uri = "s3://${var.processed_bucket_name}/evaluation-output/${local.ad_training_sfn_name_full}/" # Needs execution ID appended later

  tags = {
    Project = var.project_name
    Env     = var.env_suffix
    Purpose = "AD-Training-Workflow"
  }
}
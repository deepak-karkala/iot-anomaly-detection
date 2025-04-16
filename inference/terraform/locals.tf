locals {
  # Construct names
  alert_db_table_name_full       = lower("${var.project_name}-${var.alert_db_table_name}-${var.env_suffix}")
  inference_sfn_name_full        = "${var.project_name}-${var.inference_sfn_name}-${var.env_suffix}"
  scheduler_name_full            = "${var.project_name}-${var.scheduler_name}-${var.env_suffix}"
  lambda_get_model_func_name     = "${var.project_name}-GetApprovedModelLambda-${var.env_suffix}"
  lambda_create_sm_model_func_name = "${var.project_name}-CreateSageMakerModelLambda-${var.env_suffix}"
  lambda_process_results_func_name = "${var.project_name}-ProcessADResultsLambda-${var.env_suffix}"

  lambda_get_model_zip         = "get_approved_model_lambda.zip"
  lambda_create_sm_model_zip   = "create_sagemaker_model_lambda.zip"
  lambda_process_results_zip   = "process_inference_results_lambda.zip"

  lambda_get_model_s3_key      = "lambda-code/${local.lambda_get_model_zip}"
  lambda_create_sm_model_s3_key = "lambda-code/${local.lambda_create_sm_model_zip}"
  lambda_process_results_s3_key = "lambda-code/${local.lambda_process_results_zip}"

  # S3 Paths (base paths, execution ID etc. added dynamically)
  s3_processed_meter_uri  = "s3://${var.processed_bucket_name}/processed_meter_data/"
  s3_processed_weather_uri= "s3://${var.processed_bucket_name}/processed_weather/"
  s3_inference_feature_output_base = "s3://${var.processed_bucket_name}/inference-features/${local.inference_sfn_name_full}/"
  s3_batch_transform_output_base = "s3://${var.processed_bucket_name}/inference-output/${local.inference_sfn_name_full}/"


  tags = {
    Project = var.project_name
    Env     = var.env_suffix
    Purpose = "AD-Inference-Workflow"
  }
}
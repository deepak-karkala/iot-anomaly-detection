# --- Lambda Code Packaging ---
data "archive_file" "get_model_zip" {
  type        = "zip"
  source_dir  = var.lambda_get_model_code_dir
  output_path = "${path.module}/${local.lambda_get_model_zip}"
}
data "archive_file" "create_sm_model_zip" {
  type        = "zip"
  source_dir  = var.lambda_create_sm_model_code_dir
  output_path = "${path.module}/${local.lambda_create_sm_model_zip}"
}
data "archive_file" "process_results_zip" {
  type        = "zip"
  source_dir  = var.lambda_process_results_code_dir
  output_path = "${path.module}/${local.lambda_process_results_zip}"
}

# --- Upload Lambda Code to S3 ---
resource "aws_s3_object" "get_model_code" {
  bucket = var.scripts_bucket_name
  key    = local.lambda_get_model_s3_key
  source = data.archive_file.get_model_zip.output_path
  etag   = data.archive_file.get_model_zip.output_md5
}
resource "aws_s3_object" "create_sm_model_code" {
  bucket = var.scripts_bucket_name
  key    = local.lambda_create_sm_model_s3_key
  source = data.archive_file.create_sm_model_zip.output_path
  etag   = data.archive_file.create_sm_model_zip.output_md5
}
resource "aws_s3_object" "process_results_code" {
  bucket = var.scripts_bucket_name
  key    = local.lambda_process_results_s3_key
  source = data.archive_file.process_results_zip.output_path
  etag   = data.archive_file.process_results_zip.output_md5
}

# --- Lambda Function Definitions ---
resource "aws_lambda_function" "get_approved_model" {
  function_name = local.lambda_get_model_func_name
  role          = aws_iam_role.lambda_get_model_role.arn
  handler       = "handler.lambda_handler"
  runtime       = "python3.9"
  timeout       = 30
  memory_size   = 128
  s3_bucket     = var.scripts_bucket_name
  s3_key        = aws_s3_object.get_model_code.key
  source_code_hash = data.archive_file.get_model_zip.output_base64sha256
  tags          = local.tags
  depends_on    = [aws_iam_role_policy.lambda_get_model_policy, aws_s3_object.get_model_code]
}

resource "aws_lambda_function" "create_sagemaker_model" {
  function_name = local.lambda_create_sm_model_func_name
  role          = aws_iam_role.lambda_create_sm_model_role.arn
  handler       = "handler.lambda_handler"
  runtime       = "python3.9"
  timeout       = 60
  memory_size   = 128
  s3_bucket     = var.scripts_bucket_name
  s3_key        = aws_s3_object.create_sm_model_code.key
  source_code_hash = data.archive_file.create_sm_model_zip.output_base64sha256
  tags          = local.tags
  depends_on    = [aws_iam_role_policy.lambda_create_sm_model_policy, aws_s3_object.create_sm_model_code]
}

resource "aws_lambda_function" "process_inference_results" {
  function_name = local.lambda_process_results_func_name
  role          = aws_iam_role.lambda_process_results_role.arn
  handler       = "handler.lambda_handler"
  runtime       = "python3.9"
  timeout       = 120 # May need more time if many files/alerts
  memory_size   = 256
  s3_bucket     = var.scripts_bucket_name
  s3_key        = aws_s3_object.process_results_code.key
  source_code_hash = data.archive_file.process_results_zip.output_base64sha256
  tags          = local.tags
  environment { # Pass config via environment variables
    variables = {
      ALERT_THRESHOLD      = var.default_alert_threshold
      ALERT_DB_TABLE_NAME  = aws_dynamodb_table.alert_table.name
      AWS_DEFAULT_REGION   = var.aws_region
    }
  }
  depends_on    = [aws_iam_role_policy.lambda_process_results_policy, aws_s3_object.process_results_code, aws_dynamodb_table.alert_table]
}
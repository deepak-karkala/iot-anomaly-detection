# Package the lambda code
data "archive_file" "register_lambda_zip" {
  type        = "zip"
  source_dir  = var.lambda_code_dir
  output_path = local.lambda_zip_output_path
}

# Upload lambda zip to S3
resource "aws_s3_object" "register_lambda_code" {
  bucket = var.scripts_bucket_name # Reuse scripts bucket
  key    = local.lambda_s3_key
  source = data.archive_file.register_lambda_zip.output_path
  etag   = data.archive_file.register_lambda_zip.output_md5

  # Ensure bucket exists first (if creating scripts bucket here, add depends_on)
  # depends_on = [aws_s3_bucket.glue_scripts] # If scripts bucket is in this module
}

# Lambda Function
resource "aws_lambda_function" "register_model" {
  function_name = local.lambda_function_name_full
  role          = aws_iam_role.lambda_register_role.arn
  handler       = "handler.lambda_handler" # File.function
  runtime       = "python3.9"              # Choose appropriate runtime
  timeout       = 60                      # Seconds
  memory_size   = 256                     # MB

  s3_bucket = var.scripts_bucket_name
  s3_key    = aws_s3_object.register_lambda_code.key
  # Required if source code hash changes
  source_code_hash = data.archive_file.register_lambda_zip.output_base64sha256

  tags = local.tags

  depends_on = [
    aws_iam_role_policy.lambda_register_policy,
    aws_s3_object.register_lambda_code
  ]
}
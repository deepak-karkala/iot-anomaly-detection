data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

# --- Role for Inference Step Functions ---
resource "aws_iam_role" "step_functions_inference_role" {
  name = "${var.project_name}-sfn-ad-infer-role-${var.env_suffix}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole", Effect = "Allow",
      Principal = { Service = "states.${var.aws_region}.amazonaws.com" }
    }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy" "step_functions_inference_policy" {
  name = "${var.project_name}-sfn-ad-infer-policy-${var.env_suffix}"
  role = aws_iam_role.step_functions_inference_role.id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      { # Permissions for SageMaker, Lambda
        Effect = "Allow",
        Action = [
          "sagemaker:CreateProcessingJob",
          "sagemaker:DescribeProcessingJob",
          "sagemaker:CreateModel", # Added for CreateModel step
          "sagemaker:DescribeModel",
          "sagemaker:CreateTransformJob",
          "sagemaker:DescribeTransformJob",
          "lambda:InvokeFunction"
        ],
        Resource = "*" # Scope down if possible
      },
      { # Events permissions for .sync tasks
        Effect = "Allow",
        Action = ["events:PutTargets", "events:PutRule", "events:DescribeRule"],
        Resource = [
            "arn:${data.aws_partition.current.partition}:events:${var.aws_region}:${data.aws_caller_identity.current.account_id}:rule/StepFunctionsGetEventsForSageMakerProcessingJobsRule",
            "arn:${data.aws_partition.current.partition}:events:${var.aws_region}:${data.aws_caller_identity.current.account_id}:rule/StepFunctionsGetEventsForSageMakerTransformJobsRule"
            # Add rule for CreateModel if needed (usually fast)
            ]
      },
      { # PassRole permissions
         Effect = "Allow", Action = "iam:PassRole",
         Resource = [
            var.sagemaker_processing_role_arn, # Reuse role from training stack
            aws_iam_role.sagemaker_batch_transform_role.arn # Role defined below
            ],
         Condition = { StringEquals = { "iam:PassedToService": "sagemaker.amazonaws.com" } }
      }
    ]
  })
}


# --- Role for SageMaker Batch Transform Jobs ---
resource "aws_iam_role" "sagemaker_batch_transform_role" {
  name = "${var.project_name}-sagemaker-batch-role-${var.env_suffix}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole", Effect = "Allow", Principal = { Service = "sagemaker.amazonaws.com" }
    }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy" "sagemaker_batch_transform_policy" {
  name = "${var.project_name}-sagemaker-batch-policy-${var.env_suffix}"
  role = aws_iam_role.sagemaker_batch_transform_role.id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      { # S3 Read Features, Write Output
        Effect = "Allow",
        Action = ["s3:GetObject", "s3:ListBucket"],
        Resource = [
            "arn:aws:s3:::${var.processed_bucket_name}/inference-features/*" # Input features
            # Add S3 path for model artifact if not embedded? Usually handled by SM
            ]
      },
      {
         Effect = "Allow", Action = "s3:PutObject",
         Resource = ["arn:aws:s3:::${var.processed_bucket_name}/inference-output/*"] # Output scores
      },
      { # CloudWatch Logs
        Effect   = "Allow", Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
        Resource = "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/TransformJobs:*"
      },
      { # ECR Access (to pull inference container image - same as training)
        Effect = "Allow", Action = ["ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage", "ecr:BatchCheckLayerAvailability"],
        Resource = "*" # Needs ECR repo ARN from training stack ideally - "*" used for simplicity here
      },
      { # ECR Token
        Action = "ecr:GetAuthorizationToken", Effect = "Allow", Resource = "*"
      }
      # Add KMS if needed
    ]
  })
}

# --- Role for GetApprovedModel Lambda ---
resource "aws_iam_role" "lambda_get_model_role" {
  name = "${var.project_name}-lambda-getmodel-role-${var.env_suffix}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17", Statement = [{ Action = "sts:AssumeRole", Effect = "Allow", Principal = { Service = "lambda.amazonaws.com" }}]
  })
  tags = local.tags
}
resource "aws_iam_role_policy" "lambda_get_model_policy" {
  name = "${var.project_name}-lambda-getmodel-policy-${var.env_suffix}"
  role = aws_iam_role.lambda_get_model_role.id
  policy = jsonencode({ # Needs basic lambda exec + SageMaker describe/list
      Version = "2012-10-17", Statement = [
        { Effect = "Allow", Action = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"], Resource = "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/${local.lambda_get_model_func_name}:*" },
        { Effect = "Allow", Action = ["sagemaker:ListModelPackages", "sagemaker:DescribeModelPackage"], Resource = "arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:model-package-group/${var.ad_model_package_group_name}" } # Scope to specific group
      ]
  })
}

# --- Role for CreateSageMakerModel Lambda ---
resource "aws_iam_role" "lambda_create_sm_model_role" {
  name = "${var.project_name}-lambda-createmodel-role-${var.env_suffix}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17", Statement = [{ Action = "sts:AssumeRole", Effect = "Allow", Principal = { Service = "lambda.amazonaws.com" }}]
  })
  tags = local.tags
}
resource "aws_iam_role_policy" "lambda_create_sm_model_policy" {
  name = "${var.project_name}-lambda-createmodel-policy-${var.env_suffix}"
  role = aws_iam_role.lambda_create_sm_model_role.id
  policy = jsonencode({ # Needs basic lambda exec + SageMaker CreateModel
      Version = "2012-10-17", Statement = [
        { Effect = "Allow", Action = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"], Resource = "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/${local.lambda_create_sm_model_func_name}:*" },
        { Effect = "Allow", Action = ["sagemaker:CreateModel", "sagemaker:DescribeModelPackage"], Resource = "*" } # Needs DescribeModelPackage too potentially; Scope CreateModel if possible
      ]
  })
}

# --- Role for ProcessResults Lambda ---
resource "aws_iam_role" "lambda_process_results_role" {
  name = "${var.project_name}-lambda-procresults-role-${var.env_suffix}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17", Statement = [{ Action = "sts:AssumeRole", Effect = "Allow", Principal = { Service = "lambda.amazonaws.com" }}]
  })
  tags = local.tags
}
resource "aws_iam_role_policy" "lambda_process_results_policy" {
  name = "${var.project_name}-lambda-procresults-policy-${var.env_suffix}"
  role = aws_iam_role.lambda_process_results_role.id
  policy = jsonencode({ # Needs basic lambda exec + S3 Read + DynamoDB Write
      Version = "2012-10-17", Statement = [
        { Effect = "Allow", Action = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"], Resource = "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/${local.lambda_process_results_func_name}:*" },
        { Effect = "Allow", Action = "s3:GetObject", Resource = "arn:aws:s3:::${var.processed_bucket_name}/inference-output/*" },
        { Effect = "Allow", Action = ["dynamodb:PutItem", "dynamodb:BatchWriteItem"], Resource = aws_dynamodb_table.alert_table.arn }
      ]
  })
}

# --- Role for EventBridge Scheduler ---
resource "aws_iam_role" "scheduler_role" {
  name = "${var.project_name}-scheduler-role-${var.env_suffix}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17", Statement = [{ Action = "sts:AssumeRole", Effect = "Allow", Principal = { Service = "scheduler.amazonaws.com" }}]
  })
  tags = local.tags
}
resource "aws_iam_role_policy" "scheduler_policy" {
  name = "${var.project_name}-scheduler-policy-${var.env_suffix}"
  role = aws_iam_role.scheduler_role.id
  policy = jsonencode({ # Needs permission to start the Step Function execution
      Version = "2012-10-17", Statement = [
        { Effect = "Allow", Action = "states:StartExecution", Resource = aws_sfn_state_machine.ad_inference_state_machine.id }
      ]
  })
}
data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

# --- Role for Step Functions ---
resource "aws_iam_role" "step_functions_role" {
  name = "${var.project_name}-sfn-ad-train-role-${var.env_suffix}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole",
      Effect = "Allow",
      Principal = { Service = "states.${var.aws_region}.amazonaws.com" } # Region specific principal
    }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy" "step_functions_policy" {
  name = "${var.project_name}-sfn-ad-train-policy-${var.env_suffix}"
  role = aws_iam_role.step_functions_role.id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "sagemaker:CreateProcessingJob",
          "sagemaker:DescribeProcessingJob",
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "lambda:InvokeFunction"
        ],
        Resource = "*" # Scope down if possible, e.g., specific Lambda ARN
      },
      # Allow Step Functions to wait for SageMaker jobs (.sync integration)
      {
        Effect = "Allow",
        Action = [
          "events:PutTargets",
          "events:PutRule",
          "events:DescribeRule"
        ],
        Resource = ["arn:${data.aws_partition.current.partition}:events:${var.aws_region}:${data.aws_caller_identity.current.account_id}:rule/StepFunctionsGetEventsForSageMakerProcessingJobsRule",
                    "arn:${data.aws_partition.current.partition}:events:${var.aws_region}:${data.aws_caller_identity.current.account_id}:rule/StepFunctionsGetEventsForSageMakerTrainingJobsRule"
                   ] # Standard SageMaker integration rules
      },
       # Allow passing roles to SageMaker
      {
         Effect = "Allow",
         Action = "iam:PassRole",
         Resource = [
            aws_iam_role.sagemaker_processing_role.arn,
            aws_iam_role.sagemaker_training_role.arn
            ],
         Condition = { StringEquals = { "iam:PassedToService": "sagemaker.amazonaws.com" } }
      }
    ]
  })
}


# --- Role for SageMaker Processing Jobs ---
resource "aws_iam_role" "sagemaker_processing_role" {
  name = "${var.project_name}-sagemaker-processing-role-${var.env_suffix}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole",
      Effect = "Allow",
      Principal = { Service = "sagemaker.amazonaws.com" }
    }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy" "sagemaker_processing_policy" {
  name = "${var.project_name}-sagemaker-processing-policy-${var.env_suffix}"
  role = aws_iam_role.sagemaker_processing_role.id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      { # S3 Access (Read processed, write evaluation output, read model)
        Effect = "Allow",
        Action = ["s3:GetObject", "s3:ListBucket"],
        Resource = [
          "arn:aws:s3:::${var.processed_bucket_name}",
          "arn:aws:s3:::${var.processed_bucket_name}/*",
          "arn:aws:s3:::${var.scripts_bucket_name}", # If processing jobs need scripts/configs from S3
          "arn:aws:s3:::${var.scripts_bucket_name}/*"
          # Add model artifact bucket/prefix if reading model during evaluation
        ]
      },
      {
         Effect = "Allow",
         Action = ["s3:PutObject", "s3:DeleteObject"],
         Resource = [
            "arn:aws:s3:::${var.processed_bucket_name}/evaluation-output/*", # For evaluation report
            # Add access to offline feature store path if writing features
            "arn:aws:s3:::${var.processed_bucket_name}/feature-store-offline/*"
         ]
      },
      { # CloudWatch Logs
        Effect   = "Allow",
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
        Resource = "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/ProcessingJobs:*"
      },
      { # Feature Store Access (if engineering job writes to it)
        Effect = "Allow",
        Action = [
            "sagemaker:PutRecord",
            "sagemaker:DescribeFeatureGroup"
            # Add BatchGetRecord if reading features from FS in evaluation
        ],
        Resource = aws_sagemaker_feature_group.ad_features.arn
      }
      # Add KMS permissions if buckets/feature store are encrypted
    ]
  })
}


# --- Role for SageMaker Training Jobs ---
resource "aws_iam_role" "sagemaker_training_role" {
  name = "${var.project_name}-sagemaker-training-role-${var.env_suffix}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole",
      Effect = "Allow",
      Principal = { Service = "sagemaker.amazonaws.com" }
    }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy" "sagemaker_training_policy" {
  name = "${var.project_name}-sagemaker-training-policy-${var.env_suffix}"
  role = aws_iam_role.sagemaker_training_role.id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      { # S3 Access (Read features, write model artifacts)
        Effect = "Allow",
        Action = ["s3:GetObject", "s3:ListBucket"],
        Resource = [
          # Add S3 path for features (either FS offline store or dedicated path)
          "arn:aws:s3:::${var.processed_bucket_name}/feature-store-offline/*", # If using FS
          "arn:aws:s3:::${var.processed_bucket_name}/ad-features/*", # If using dedicated feature path
          "arn:aws:s3:::${var.scripts_bucket_name}", # If training needs scripts/configs
          "arn:aws:s3:::${var.scripts_bucket_name}/*"
        ]
      },
      {
         Effect = "Allow",
         Action = ["s3:PutObject"],
         Resource = [ "arn:aws:s3:::${var.processed_bucket_name}/model-artifacts/*" ] # Or dedicated model bucket
      },
      { # CloudWatch Logs
        Effect   = "Allow",
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
        Resource = "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/TrainingJobs:*"
      },
      { # ECR Access (to pull training container image)
        Effect = "Allow",
        Action = [
            "ecr:GetDownloadUrlForLayer",
            "ecr:BatchGetImage",
            "ecr:BatchCheckLayerAvailability"
            # Potentially ecr:GetAuthorizationToken depending on setup
        ],
        Resource = aws_ecr_repository.ad_training_repo.arn
      },
      { # ECR Authorization Token (Needed by SageMaker)
        Action = "ecr:GetAuthorizationToken",
        Effect = "Allow",
        Resource = "*"
      }
      # Add KMS permissions if needed
      # Add Feature Store GetRecord permissions if reading features directly
    ]
  })
}


# --- Role for Lambda Function (Register Model) ---
resource "aws_iam_role" "lambda_register_role" {
  name = "${var.project_name}-lambda-register-role-${var.env_suffix}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole",
      Effect = "Allow",
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy" "lambda_register_policy" {
  name = "${var.project_name}-lambda-register-policy-${var.env_suffix}"
  role = aws_iam_role.lambda_register_role.id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      { # Basic Lambda execution policy
        Effect = "Allow",
        Action = [
            "logs:CreateLogGroup",
            "logs:CreateLogStream",
            "logs:PutLogEvents"
            ],
        Resource = "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/${local.lambda_function_name_full}:*"
      },
      { # SageMaker Model Registry access
        Effect = "Allow",
        Action = [
            "sagemaker:CreateModelPackage",
            "sagemaker:DescribeModelPackageGroup" # Needed to check status/existence maybe
            ],
        Resource = aws_sagemaker_model_package_group.ad_model_group.arn
      },
      { # S3 read access for evaluation report
         Effect = "Allow",
         Action = "s3:GetObject",
         Resource = "arn:aws:s3:::${var.processed_bucket_name}/evaluation-output/*"
      }
      # Add KMS permissions if needed
    ]
  })
}
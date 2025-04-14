output "ad_feature_group_arn" {
  description = "ARN of the AD SageMaker Feature Group"
  value       = aws_sagemaker_feature_group.ad_features.arn
}

output "ecr_repository_url" {
  description = "URL of the ECR repository for the training container"
  value       = aws_ecr_repository.ad_training_repo.repository_url
}

output "ad_model_package_group_arn" {
  description = "ARN of the AD SageMaker Model Package Group"
  value       = aws_sagemaker_model_package_group.ad_model_group.arn
}

output "register_model_lambda_arn" {
  description = "ARN of the model registration Lambda function"
  value       = aws_lambda_function.register_model.arn
}

output "ad_training_state_machine_arn" {
  description = "ARN of the AD training Step Functions State Machine"
  value       = aws_sfn_state_machine.ad_training_state_machine.id # Use id for ARN
}

output "sagemaker_processing_role_arn" {
  description = "ARN of the IAM role for SageMaker Processing Jobs"
  value       = aws_iam_role.sagemaker_processing_role.arn
}

output "sagemaker_training_role_arn" {
  description = "ARN of the IAM role for SageMaker Training Jobs"
  value       = aws_iam_role.sagemaker_training_role.arn
}
output "ad_alert_dynamodb_table_name" {
  description = "Name of the DynamoDB table storing AD alerts"
  value       = aws_dynamodb_table.alert_table.name
}

output "ad_inference_state_machine_arn" {
  description = "ARN of the AD inference Step Functions State Machine"
  value       = aws_sfn_state_machine.ad_inference_state_machine.id
}

output "daily_inference_scheduler_name" {
  description = "Name of the EventBridge Scheduler for daily inference"
  value       = aws_scheduler_schedule.daily_inference_trigger.name
}

output "sagemaker_batch_transform_role_arn" {
  description = "ARN of the IAM role for SageMaker Batch Transform Jobs"
  value       = aws_iam_role.sagemaker_batch_transform_role.arn
}
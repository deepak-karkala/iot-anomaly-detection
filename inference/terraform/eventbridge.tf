resource "aws_scheduler_schedule" "daily_inference_trigger" {
  name       = local.scheduler_name_full
  group_name = "default" # Or create a custom group

  flexible_time_window {
    mode = "OFF" # Run exactly at schedule time
  }

  schedule_expression          = var.scheduler_expression
  schedule_expression_timezone = "UTC" # Specify timezone

  target {
    arn      = aws_sfn_state_machine.ad_inference_state_machine.id
    role_arn = aws_iam_role.scheduler_role.arn

    # Input for the Step Function execution (can be dynamic)
    # Example: Calculate yesterday's date, or use Step Functions context objects if triggered differently
    input = jsonencode({
      "inference_date" = time_static.yesterday.id # Requires time_static resource or Lambda to calculate
      # Or pass static values, or leave empty if SFN calculates date
      # Example static: "inference_date" = "2024-10-27"
      # Example placeholder: "inference_date.$" = "$$.Execution.StartTime" # If SFN input is StartTime
    })
  }

  depends_on = [
    aws_sfn_state_machine.ad_inference_state_machine,
    aws_iam_role_policy.scheduler_policy
  ]
}

# Optional: Resource to get yesterday's date for scheduler input
# resource "time_static" "yesterday" {
#   rfc3339 = timestamp() # Needs Terraform 0.12+ timestamp()
#   triggers = {
#      # Re-evaluate daily - may need external trigger or manual update for true daily date
#      # This approach might not work reliably for dynamic daily date in scheduler input
#      # Better approach: Have Step Function calculate the date or use Lambda trigger
#   }
# }

# NOTE: Passing dynamic dates like "yesterday" directly into the scheduler input is tricky.
# Often, the Step Function itself calculates the date based on execution time,
# or an initial Lambda state calculates it. For simplicity, the input is shown
# statically or using a placeholder here. You'll need to adapt the trigger/input mechanism.
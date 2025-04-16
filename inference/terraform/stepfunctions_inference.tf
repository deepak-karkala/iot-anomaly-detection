resource "aws_sfn_state_machine" "ad_inference_state_machine" {
  name     = local.inference_sfn_name_full
  role_arn = aws_iam_role.step_functions_inference_role.arn
  tags     = local.tags

  definition = jsonencode({
    Comment = "AD Batch Inference Workflow: Get Model -> Create Model Resource -> Feature Eng -> Batch Transform -> Process Results"
    StartAt = "GetApprovedModelPackage"

    States = {
      GetApprovedModelPackage = {
        Type = "Task",
        Resource = "arn:aws:states:::lambda:invoke",
        Parameters = {
          FunctionName = aws_lambda_function.get_approved_model.function_name,
          Payload = {
             "ModelPackageGroupName" = var.ad_model_package_group_name
          }
        },
        ResultSelector = { # Extract only the needed ARN from Lambda output
           "ModelPackageArn.$": "$.Payload.ModelPackageArn"
        },
        ResultPath = "$.approved_model", # Store ARN in execution state
        Retry = [{ ErrorEquals = ["Lambda.TooManyRequestsException", "Lambda.ServiceException"], IntervalSeconds=3, MaxAttempts=3, BackoffRate=1.5 }],
        Catch = [{ ErrorEquals = ["States.ALL"], Next = "WorkflowFailed", ResultPath = "$.errorInfo" }],
        Next = "CreateModelResource"
      },

      CreateModelResource = {
         Type = "Task",
         Resource = "arn:aws:states:::lambda:invoke",
         Parameters = {
            FunctionName = aws_lambda_function.create_sagemaker_model.function_name,
            Payload = {
                "ModelPackageArn.$": "$.approved_model.ModelPackageArn",
                "ModelNamePrefix": "${local.inference_sfn_name_full}-model", # Prefix for unique model name
                "ExecutionRoleArn": aws_iam_role.sagemaker_batch_transform_role.arn # Role the SM Model resource will use
            }
         },
         ResultSelector = {
             "ModelName.$": "$.Payload.ModelName"
         },
         ResultPath = "$.sagemaker_model",
         Retry = [{ ErrorEquals = ["Lambda.TooManyRequestsException", "Lambda.ServiceException"], IntervalSeconds=3, MaxAttempts=3, BackoffRate=1.5 }],
         Catch = [{ ErrorEquals = ["States.ALL"], Next = "WorkflowFailed", ResultPath = "$.errorInfo" }],
         Next = "FeatureEngineeringInference"
      },

      FeatureEngineeringInference = {
        Type = "Task",
        Resource = "arn:aws:states:::sagemaker:createProcessingJob.sync",
        Parameters = {
          ProcessingJobName.$ = "States.Format('InferFeatEng-{}-{}', $$.StateMachine.Name, $$.Execution.Id)", # Use Execution.Id
          ProcessingResources = { # Use smaller instance potentially
            ClusterConfig = { InstanceCount = 1, InstanceType = "ml.m5.large", VolumeSizeInGB = 20 }
          },
          AppSpecification = {
            ImageUri = var.spark_processing_image_uri,
            ContainerArguments = [
               # Calculate inference date dynamically (e.g., yesterday based on SFN start time)
               # This requires more complex state manipulation or a preceding Lambda state.
               # Placeholder: Assume inference_date is passed in SFN execution input.
               "--inference-date.$", "$.inference_date"
            ],
             ContainerEntrypoint = ["python3", "/opt/ml/processing/input/code/feature_engineering_inference.py"]
          },
          ProcessingInputs = [
             { InputName = "code", S3Input = { S3Uri = "s3://${var.scripts_bucket_name}/scripts/feature_engineering_inference.py", LocalPath = "/opt/ml/processing/input/code/", S3DataType = "S3Prefix", S3InputMode = "File"}},
             { InputName = "processed_meter", S3Input = { S3Uri = local.s3_processed_meter_uri, LocalPath = "/opt/ml/processing/input/processed_meter/", S3DataType = "S3Prefix", S3InputMode = "File"}},
             { InputName = "processed_weather", S3Input = { S3Uri = local.s3_processed_weather_uri, LocalPath = "/opt/ml/processing/input/processed_weather/", S3DataType = "S3Prefix", S3InputMode = "File"}}
          ],
          ProcessingOutputConfig = { Outputs = [{ # Output inference features
              OutputName = "inference_features",
              S3Output = { S3Uri.$ = "States.Format('{}{}/features', local.s3_inference_feature_output_base, $$.Execution.Name)", LocalPath = "/opt/ml/processing/output/inference_features/", S3UploadMode = "EndOfJob"}
            }]
          },
          RoleArn = var.sagemaker_processing_role_arn
        },
        ResultPath = "$.feature_eng_output",
        Retry = [{ ErrorEquals = ["States.ALL"], IntervalSeconds = 10, MaxAttempts = 2, BackoffRate = 2.0 }],
        Catch = [{ ErrorEquals = ["States.ALL"], Next = "WorkflowFailed", ResultPath = "$.errorInfo" }],
        Next = "BatchTransform"
      },

      BatchTransform = {
        Type = "Task",
        Resource = "arn:aws:states:::sagemaker:createTransformJob.sync",
        Parameters = {
            TransformJobName.$ = "States.Format('AD-Inference-{}-{}', $$.StateMachine.Name, $$.Execution.Id)",
            ModelName.$ = "$.sagemaker_model.ModelName", # Use model name created in previous step
            TransformInput = {
                DataSource = { S3DataSource = { S3DataType = "S3Prefix", S3Uri.$ = "$.feature_eng_output.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri" } },
                ContentType = "text/csv", # Assuming feature eng outputs CSV
                SplitType = "Line",
                CompressionType = "None"
            },
            TransformOutput = {
                S3OutputPath.$ = "States.Format('{}{}/scores', local.s3_batch_transform_output_base, $$.Execution.Name)",
                Accept = "text/csv", # Match output_fn in inference.py
                AssembleWith = "Line"
            },
            TransformResources = {
                InstanceCount = var.batch_transform_instance_count,
                InstanceType = var.batch_transform_instance_type
            }
            # Add DataProcessing, Environment if needed
        },
        ResultPath = "$.batch_transform_output",
        Retry = [{ ErrorEquals = ["States.ALL"], IntervalSeconds = 30, MaxAttempts = 2, BackoffRate = 2.0 }],
        Catch = [{ ErrorEquals = ["States.ALL"], Next = "WorkflowFailed", ResultPath = "$.errorInfo" }],
        Next = "ProcessResults"
      },

      ProcessResults = {
        Type = "Task",
        Resource = "arn:aws:states:::lambda:invoke",
        Parameters = {
            FunctionName = aws_lambda_function.process_inference_results.function_name,
            # Lambda needs to know which S3 output path contains the scores
            # Assuming Batch Transform output URI is directly usable
            # Note: Lambda triggered by S3 event is an alternative pattern
            Payload = {
               "S3OutputPath.$" = "$.batch_transform_output.TransformOutput.S3OutputPath"
               # Pass other context if needed
            }
        },
        Retry = [{ ErrorEquals = ["Lambda.TooManyRequestsException", "Lambda.ServiceException"], IntervalSeconds=5, MaxAttempts=3, BackoffRate=2.0 }],
        Catch = [{ ErrorEquals = ["States.ALL"], Next = "WorkflowFailed", ResultPath = "$.errorInfo" }],
        Next = "WorkflowSucceeded"
      },

      WorkflowFailed = { Type = "Fail", Cause = "Inference workflow failed", Error = "WorkflowError" },
      WorkflowSucceeded = { Type = "Succeed" }
    }
  })
  depends_on = [ # Ensure all dependencies are created
    aws_lambda_function.get_approved_model,
    aws_lambda_function.create_sagemaker_model,
    aws_lambda_function.process_inference_results,
    aws_iam_role.step_functions_inference_role,
    aws_iam_role.sagemaker_batch_transform_role
    # Add Processing Role if it's different from training one
  ]
}
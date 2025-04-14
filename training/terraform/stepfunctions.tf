resource "aws_sfn_state_machine" "ad_training_state_machine" {
  name     = local.ad_training_sfn_name_full
  role_arn = aws_iam_role.step_functions_role.arn
  tags     = local.tags

  # Definition using Amazon States Language (ASL)
  # Needs careful construction to pass parameters between steps
  definition = jsonencode({
    Comment = "AD Model Training Workflow: Feature Eng -> Train -> Evaluate -> Register (Pending Approval)"
    StartAt = "ValidateSchema" # Or start directly at FeatureEngineering if schema validation is separate

    States = {
      ValidateSchema = {
        Type = "Task",
        Resource = "arn:aws:states:::sagemaker:createProcessingJob.sync", # .sync waits for completion
        Parameters = {
          ProcessingJobName.$ = "States.Format('ValidateSchema-{}-{}', $$.StateMachine.Name, $$.Execution.Name)",
          ProcessingResources = {
            ClusterConfig = {
              InstanceCount = 1 # Schema validation usually needs minimal resources
              InstanceType = "ml.t3.medium" # Can use smaller instance
              VolumeSizeInGB = 10
            }
          },
          AppSpecification = {
            ImageUri = "YOUR_SPARK_PROCESSING_IMAGE_URI" # Use a standard SageMaker Spark Processing image URI
            # Example: 763104351884.dkr.ecr.us-east-1.amazonaws.com/spark-processing:3.1-cpu-py37-v1.0
            # Replace with correct image for your region and desired Spark/Python version
            ContainerArguments = [
               "--data-path", local.s3_processed_meter_uri, # Path to check
               "--expected-schema", "{ \"columns\": [ { \"name\": \"event_ts\", \"type\": \"timestamp\" }, ... ] }" # Pass expected schema JSON string - replace ...
               # Add date range args if needed: "--start-date.$": "$.data_params.start_date"
            ],
            ContainerEntrypoint = ["python3", "/opt/ml/processing/input/code/validate_schema.py"] # Path inside container
          },
          ProcessingInputs = [{ # Mount the script code
            InputName = "code",
            S3Input = {
              S3Uri = "s3://${var.scripts_bucket_name}/scripts/validate_schema.py" # Assuming script is directly in bucket/scripts/
              LocalPath = "/opt/ml/processing/input/code/"
              S3DataType = "S3Prefix"
              S3InputMode = "File"
            }
          }],
          RoleArn = aws_iam_role.sagemaker_processing_role.arn
        },
        Retry = [{ # Optional retry logic
             ErrorEquals = ["States.ALL"],
             IntervalSeconds = 10,
             MaxAttempts = 2,
             BackoffRate = 2.0
          }],
        Catch = [{ # Catch errors and transition to a Fail state
            ErrorEquals = ["States.ALL"],
            Next = "WorkflowFailed",
            ResultPath = "$.errorInfo" # Capture error info
          }],
        Next = "FeatureEngineering"
      },

      FeatureEngineering = {
        Type = "Task",
        Resource = "arn:aws:states:::sagemaker:createProcessingJob.sync",
        Parameters = {
          ProcessingJobName.$ = "States.Format('FeatureEng-{}-{}', $$.StateMachine.Name, $$.Execution.Name)",
          ProcessingResources = {
            ClusterConfig = {
              InstanceCount = var.processing_instance_count,
              InstanceType = var.processing_instance_type,
              VolumeSizeInGB = 30 # Adjust as needed
            }
          },
          AppSpecification = {
            ImageUri = "YOUR_SPARK_PROCESSING_IMAGE_URI" # Use same/similar Spark image
            ContainerArguments = [
               "--meter-data-path", local.s3_processed_meter_uri,
               "--weather-data-path", "s3://${var.processed_bucket_name}/processed_weather/", # ASSUME processed weather path exists
               "--start-date.$", "$.data_params.start_date", # Use input parameters
               "--end-date.$", "$.data_params.end_date",
               "--feature-group-name", local.ad_feature_group_name_full
            ],
            ContainerEntrypoint = ["python3", "/opt/ml/processing/input/code/feature_engineering.py"]
          },
          ProcessingInputs = [{
            InputName = "code",
            S3Input = {
              S3Uri = "s3://${var.scripts_bucket_name}/scripts/feature_engineering.py"
              LocalPath = "/opt/ml/processing/input/code/"
              S3DataType = "S3Prefix"
              S3InputMode = "File"
            }
          }],
          RoleArn = aws_iam_role.sagemaker_processing_role.arn
        },
         Retry = [{ ErrorEquals = ["States.ALL"], IntervalSeconds = 10, MaxAttempts = 2, BackoffRate = 2.0 }],
         Catch = [{ ErrorEquals = ["States.ALL"], Next = "WorkflowFailed", ResultPath = "$.errorInfo" }],
        Next = "ModelTraining"
      },

      ModelTraining = {
        Type = "Task",
        Resource = "arn:aws:states:::sagemaker:createTrainingJob.sync",
        Parameters = {
          TrainingJobName.$ = "States.Format('AD-Training-{}-{}', $$.StateMachine.Name, $$.Execution.Name)",
          AlgorithmSpecification = {
            TrainingImage = var.training_image_uri, # The custom container image
            TrainingInputMode = "File"
          },
          HyperParameters = { # Pass hyperparameters from input
            "model-strategy.$" = "$.training_params.model_strategy", # e.g., "LR_LOF"
            "lof-neighbors.$" = "States.Format('{}', $.training_params.lof_neighbors)", # Convert numbers to strings for HyperParameters
            "lof-contamination.$" = "$.training_params.lof_contamination",
            "feature-columns" = "daily_energy_kwh,avg_temp_diff,hdd,avg_temp_c,energy_lag_1d,energy_roll_avg_7d", # Pass as string - REPLACE with actual list
            "git-hash.$" = "$.git_hash",
             # Add other hyperparameters...
             "sagemaker_program" = "train.py", # If using framework container
             "sagemaker_submit_directory" = "s3://${var.scripts_bucket_name}/scripts/train/" # If using framework container & need source dir
          },
          InputDataConfig = [{
            ChannelName = "features", # Matches directory /opt/ml/input/data/features
            DataSource = {
              S3DataSource = {
                S3DataType = "S3Prefix",
                # Read features from FS Offline Store or dedicated path
                S3Uri = local.s3_offline_store_uri # Needs filtering/querying for dates - complex here.
                # Alternative: Have Feature Eng step output to a unique S3 path per execution
                # S3Uri.$ = "$.feature_engineering_output_path" # If passed from previous step
              }
            },
            ContentType = "application/x-parquet", # Adjust if different
            CompressionType = "None"
          }],
          OutputDataConfig = {
            # SageMaker automatically creates S3 path for model.tar.gz
            # Specify a base path if desired, otherwise uses default SageMaker bucket
            S3OutputPath = "s3://${var.processed_bucket_name}/model-artifacts/"
          },
          ResourceConfig = {
            InstanceCount = var.training_instance_count,
            InstanceType = var.training_instance_type,
            VolumeSizeInGB = 50 # Adjust as needed
          },
          StoppingCondition = { MaxRuntimeInSeconds = 3600 }, # 1 hour limit example
          RoleArn = aws_iam_role.sagemaker_training_role.arn
        },
        ResultPath = "$.training_job_output", # Store output of training job (includes model artifact S3 URI)
        Retry = [{ ErrorEquals = ["States.ALL"], IntervalSeconds = 30, MaxAttempts = 2, BackoffRate = 2.0 }],
        Catch = [{ ErrorEquals = ["States.ALL"], Next = "WorkflowFailed", ResultPath = "$.errorInfo" }],
        Next = "ModelEvaluation"
      },

      ModelEvaluation = {
        Type = "Task",
        Resource = "arn:aws:states:::sagemaker:createProcessingJob.sync",
        Parameters = {
          ProcessingJobName.$ = "States.Format('ModelEval-{}-{}', $$.StateMachine.Name, $$.Execution.Name)",
          ProcessingResources = { # Similar to Feature Eng job
            ClusterConfig = { InstanceCount = var.processing_instance_count, InstanceType = var.processing_instance_type, VolumeSizeInGB = 30 }
          },
          AppSpecification = {
            ImageUri = "YOUR_SKLEARN_PROCESSING_IMAGE_URI", # Use SageMaker Scikit-learn/Python image
            # Example: 683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3
             ContainerArguments = [
               # Pass historical labels URI if available (from input?)
               # "--historical-labels-s3-uri", "s3://...",
               # Pass training job metadata for throughput calculation
               "--training-duration-seconds.$", "$.training_job_output.TrainingTimeInSeconds",
               "--training-record-count", "10000" # Placeholder - Need actual count from training/features
            ],
            ContainerEntrypoint = ["python3", "/opt/ml/processing/input/code/evaluate.py"]
          },
          ProcessingInputs = [
            { # Mount the evaluation script
              InputName = "code",
              S3Input = { S3Uri = "s3://${var.scripts_bucket_name}/scripts/evaluate.py", LocalPath = "/opt/ml/processing/input/code/", S3DataType = "S3Prefix", S3InputMode = "File"}
            },
            { # Mount the trained model artifact
              InputName = "model",
              S3Input = { S3Uri.$ = "$.training_job_output.ModelArtifacts.S3ModelArtifacts", LocalPath = "/opt/ml/processing/model/", S3DataType = "S3Prefix", S3InputMode = "File"}
            },
            { # Mount the evaluation features data (hold-out set)
              InputName = "eval_features",
              S3Input = { S3Uri = local.s3_offline_store_uri , LocalPath = "/opt/ml/processing/input/eval_features/", S3DataType = "S3Prefix", S3InputMode = "File" } # Needs date filtering/specific eval set path
            }
            # Add input for historical labels if using
          ],
          ProcessingOutputConfig = { # Define where evaluation report goes
            Outputs = [{
              OutputName = "evaluation_report",
              S3Output = {
                 S3Uri = local.s3_evaluation_output_uri, # Base path
                 LocalPath = "/opt/ml/processing/evaluation/", # Path inside container
                 S3UploadMode = "EndOfJob"
              }
            }]
          },
          RoleArn = aws_iam_role.sagemaker_processing_role.arn
        },
        ResultPath = "$.evaluation_job_output", # Store output of evaluation job (includes S3 URI of report)
        Retry = [{ ErrorEquals = ["States.ALL"], IntervalSeconds = 10, MaxAttempts = 2, BackoffRate = 2.0 }],
        Catch = [{ ErrorEquals = ["States.ALL"], Next = "WorkflowFailed", ResultPath = "$.errorInfo" }],
        Next = "CheckEvaluation"
      },

      CheckEvaluation = {
        Type = "Choice",
        Choices = [
          { # Check if evaluation output contains an error or metrics are below threshold
            # This requires parsing the evaluation_report.json - Complex directly in Choice state
            # Option A: Lambda function to parse report and return simple status (Pass/Fail)
            # Option B: Check for presence of S3 output file (less robust)
            # Let's assume Option A (needs another Lambda) - Placeholder logic:
            Variable = "$.evaluation_job_output.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri", # Check if output exists
            IsPresent = true,
            Next = "RegisterModelLambda" # If output exists (simplistic check)
          }
          # Add more complex checks here based on parsed metrics if using Lambda check
        ],
        Default = "EvaluationFailed" # Go to fail state if checks fail
      },

      RegisterModelLambda = {
        Type = "Task",
        Resource = "arn:aws:states:::lambda:invoke",
        Parameters = {
          FunctionName = aws_lambda_function.register_model.function_name,
          Payload = { # Construct payload for the Lambda function
            "model_artifact_url.$" = "$.training_job_output.ModelArtifacts.S3ModelArtifacts",
            "evaluation_report_url.$" = "States.Format('{}/evaluation_report.json', $.evaluation_job_output.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri)",
            "model_package_group_name" = local.ad_model_package_group_name_full,
            "git_hash.$" = "$.git_hash",
            "feature_group_name" = local.ad_feature_group_name_full,
            "training_params.$" = "$.training_params",
            "data_params.$" = "$.data_params",
            "image_uri" = var.training_image_uri
          }
        },
         Retry = [{ ErrorEquals = ["Lambda.ServiceException", "Lambda.AWSLambdaException", "Lambda.SdkClientException"], IntervalSeconds = 2, MaxAttempts = 3, BackoffRate = 2.0 }],
         Catch = [{ ErrorEquals = ["States.ALL"], Next = "WorkflowFailed", ResultPath = "$.errorInfo" }],
         ResultPath = "$.registration_output",
         Next = "WorkflowSucceeded"
      },

      EvaluationFailed = { # State if evaluation check fails
        Type = "Fail",
        Cause = "Model evaluation metrics or throughput did not meet threshold",
        Error = "EvaluationFailed"
      },

      WorkflowFailed = { # Generic Fail state
        Type = "Fail",
        Cause.$ = "$.errorInfo.Cause", # Pass error info
        Error.$ = "$.errorInfo.Error"
      },

      WorkflowSucceeded = {
        Type = "Succeed"
      }
    }
  })

  # Depends on the Lambda function and roles being created
  depends_on = [
    aws_lambda_function.register_model,
    aws_iam_role.step_functions_role,
    aws_iam_role.sagemaker_processing_role,
    aws_iam_role.sagemaker_training_role
  ]
}
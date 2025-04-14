# SageMaker Feature Group for AD Features
resource "aws_sagemaker_feature_group" "ad_features" {
  feature_group_name = local.ad_feature_group_name_full
  record_identifier_name = "apartment_record_id" # Must match feature eng script output
  event_time_feature_name = "event_time"         # Must match feature eng script output
  role_arn = aws_iam_role.sagemaker_processing_role.arn # Role to access S3 for offline store

  # Define features EXACTLY matching the output DataFrame schema of feature_engineering.py
  feature_definition { feature_name = "apartment_record_id" ; feature_type = "String" }
  feature_definition { feature_name = "event_time" ; feature_type = "Fractional" } # Fractional for timestamp ms precision
  feature_definition { feature_name = "event_date" ; feature_type = "String" } # Often stored as String in FS, use date type if supported/needed
  feature_definition { feature_name = "building_id" ; feature_type = "String" }
  feature_definition { feature_name = "daily_energy_kwh" ; feature_type = "Fractional" }
  feature_definition { feature_name = "avg_temp_diff" ; feature_type = "Fractional" }
  feature_definition { feature_name = "daily_water_l" ; feature_type = "Fractional" }
  feature_definition { feature_name = "hdd" ; feature_type = "Fractional" }
  feature_definition { feature_name = "avg_temp_c" ; feature_type = "Fractional" }
  feature_definition { feature_name = "energy_lag_1d" ; feature_type = "Fractional" }
  feature_definition { feature_name = "energy_roll_avg_7d" ; feature_type = "Fractional" }
  feature_definition { feature_name = "temp_diff_lag_1d" ; feature_type = "Fractional" }
  # Add ALL other feature definitions...

  # Disable online store if only batch inference is needed initially
  online_store_config {
    enable_online_store = false
  }

  # Configure offline store (S3)
  offline_store_config {
    s3_storage_config {
      s3_uri = local.s3_offline_store_uri # Defined in locals.tf
      # Add kms_key_id if using encryption
    }
    disable_glue_table_creation = false # Automatically create Glue table for offline store
    data_format                 = "Parquet"
  }

  tags = local.tags

  # Ensure the role exists before creating the feature group
  depends_on = [aws_iam_role.sagemaker_processing_role]
}


# SageMaker Model Package Group
resource "aws_sagemaker_model_package_group" "ad_model_group" {
  model_package_group_name = local.ad_model_package_group_name_full
  model_package_group_description = "Model Package Group for Apartment Anomaly Detection models (${var.env_suffix})"
  tags = local.tags
}
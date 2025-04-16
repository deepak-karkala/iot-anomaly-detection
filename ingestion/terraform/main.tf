/*
- Defines variables for customization.
- Creates two S3 buckets (raw, processed). 
- Creates an IAM Role (glue_service_role) that the Glue service can assume.
- Attaches the managed AWSGlueServiceRole policy and a custom policy granting 
  necessary S3 read/write access to the specific buckets (and the script bucket).
- Creates the Glue Data Catalog database.
- Creates the Glue Catalog table (processed_meter_data) defining the expected
  schema after processing, including data types and partition keys. This table
  definition tells Glue and Athena how to interpret the data in the processed S3 location.
- Creates the Glue Job (ingest_meter_data_job), referencing the script location
  in S3, the IAM role, Glue version, worker configuration, and passing necessary
  parameters as default arguments. It depends_on the table and policy being created first.
- Outputs key resource names/ARNs.
*/


terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}


# --- Locals ---
# Use locals to construct unique bucket names
locals {
  raw_bucket_name       = lower("${var.project_name}-raw-data-${var.env_suffix}")
  processed_bucket_name = lower("${var.project_name}-processed-data-${var.env_suffix}")
  scripts_bucket_name   = lower("${var.project_name}-glue-scripts-${var.env_suffix}")
  glue_script_s3_path   = "s3://${local.scripts_bucket_name}/${var.glue_script_s3_key}"
}


# --- Resources ---

# S3 Buckets
resource "aws_s3_bucket" "raw_data" {
  bucket = local.raw_bucket_name
  tags = {
    Name    = "${var.project_name}-raw-data-${var.env_suffix}"
    Project = var.project_name
    Env     = var.env_suffix
  }
}

resource "aws_s3_bucket" "processed_data" {
  bucket = local.processed_bucket_name
  tags = {
    Name    = "${var.project_name}-processed-data-${var.env_suffix}"
    Project = var.project_name
    Env     = var.env_suffix
  }
}

# S3 Bucket for Glue Scripts
resource "aws_s3_bucket" "glue_scripts" {
  bucket = local.scripts_bucket_name
  tags = {
    Name    = "${var.project_name}-glue-scripts-${var.env_suffix}"
    Project = var.project_name
    Env     = var.env_suffix
  }
}

resource "aws_s3_object" "glue_script_upload" {
  bucket = aws_s3_bucket.glue_scripts.id
  key    = var.glue_script_s3_key
  source = var.glue_script_local_path
  etag   = filemd5(var.glue_script_local_path) # Ensures upload on change
}


# IAM Role for Glue
resource "aws_iam_role" "glue_service_role" {
  name = "${var.project_name}-glue-service-role-${var.env_suffix}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "glue.amazonaws.com"
        }
      },
    ]
  })

  tags = {
    Name    = "${var.project_name}-glue-role-${var.env_suffix}"
    Project = var.project_name
    Env     = var.env_suffix
  }
}

# Attach necessary policies
resource "aws_iam_role_policy_attachment" "glue_service_policy" {
  role       = aws_iam_role.glue_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
}

# Add S3 access policy
resource "aws_iam_role_policy" "glue_s3_access" {
  name = "${var.project_name}-glue-s3-policy-${var.env_suffix}"
  role = aws_iam_role.glue_service_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Effect   = "Allow"
        Resource = [
          aws_s3_bucket.raw_data.arn,
          "${aws_s3_bucket.raw_data.arn}/*",
          aws_s3_bucket.glue_scripts.arn,
          "${aws_s3_bucket.glue_scripts.arn}/*"
        ]
      },
      {
        Action = [
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Effect   = "Allow"
        Resource = [
          aws_s3_bucket.processed_data.arn,
          "${aws_s3_bucket.processed_data.arn}/*"
        ]
      },
    ]
  })
}


# Glue Data Catalog Database
resource "aws_glue_catalog_database" "hometech_db" {
  name = var.glue_catalog_db_name
}

# Glue Data Catalog Table for Processed Meter Data
resource "aws_glue_catalog_table" "processed_meter_data_table" {
  name          = var.processed_meter_table_name
  database_name = aws_glue_catalog_database.hometech_db.name

  table_type = "EXTERNAL_TABLE"

  parameters = {
    "EXTERNAL"            = "TRUE"
    "parquet.compression" = "SNAPPY"
    "classification"      = "parquet"
  }

  storage_descriptor {
    # Construct location using variables and locals
    location      = "s3://${local.processed_bucket_name}/${var.processed_meter_table_name}/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"

    ser_de_info {
      name                  = "processed-meter-serde"
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
      parameters = {
        "serialization.format" = "1"
      }
    }

    # Define columns matching the output of the Glue script
    columns = [
      { name = "event_ts", type = "timestamp", comment = "Timestamp of the meter reading" },
      { name = "meter_id", type = "string", comment = "Unique identifier for the meter" },
      { name = "apartment_id", type = "string", comment = "Identifier for the apartment" },
      { name = "building_id", type = "string", comment = "Identifier for the building" },
      { name = "energy_kwh", type = "double", comment = "Energy consumption reading" },
      { name = "room_temp_c", type = "double", comment = "Room temperature reading" },
      { name = "setpoint_temp_c", type = "double", comment = "Setpoint temperature" },
      { name = "hot_water_litres", type = "double", comment = "Hot water consumption" }
    ]
  }

  # Define partition keys matching the Glue script output and S3 structure
  partition_keys = [
    { name = "year", type = "int", comment = "Partition key: Year of event" },
    { name = "month", type = "int", comment = "Partition key: Month of event" },
    { name = "day", type = "int", comment = "Partition key: Day of event" }
  ]
}

# AWS Glue Job
resource "aws_glue_job" "ingest_meter_data_job" {
  name         = "${var.glue_job_name}-${var.env_suffix}"
  role_arn     = aws_iam_role.glue_service_role.arn
  glue_version = "4.0"
  worker_type  = "G.1X"
  number_of_workers = 5

  command {
    script_location = local.glue_script_s3_path
    python_version  = "3"
  }

  default_arguments = {
    "--job-language"         = "python"
    "--job-bookmark-option"  = "job-bookmark-enable"
    "--enable-metrics"       = ""
    # Pass required arguments to the script (using locals/vars)
    "--source_path"          = "s3://${local.raw_bucket_name}/meter/"
    "--destination_path"     = "s3://${local.processed_bucket_name}/${var.processed_meter_table_name}/"
    "--database_name"        = aws_glue_catalog_database.hometech_db.name
    "--table_name"           = aws_glue_catalog_table.processed_meter_data_table.name
  }

  tags = {
    Name    = "${var.glue_job_name}-${var.env_suffix}"
    Project = var.project_name
    Env     = var.env_suffix
  }

  # Ensure script is uploaded before job is created/updated
  depends_on = [
    aws_iam_role_policy.glue_s3_access,
    aws_glue_catalog_table.processed_meter_data_table,
    aws_s3_object.glue_script_upload
  ]
}


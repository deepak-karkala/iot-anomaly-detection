# --- Variables ---
variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "eu-central-1"
}

variable "project_name" {
  description = "Base name for resources (e.g., hometech-ml)"
  type        = string
  default     = "hometech-ml"
}

variable "env_suffix" {
  description = "Environment suffix (e.g., dev, prod, or unique id)"
  type        = string
  default     = "dev-unique-suffix"
}


variable "raw_bucket_name" {
  description = "Name for the raw data S3 bucket"
  type        = string
  default     = "" # Calculated below
}

variable "processed_bucket_name" {
  description = "Name for the processed data S3 bucket"
  type        = string
  default     = "" # Calculated below
}

variable "scripts_bucket_name" {
  description = "Name for the S3 bucket storing Glue scripts"
  type        = string
  default     = "" # Calculated below
}


variable "glue_catalog_db_name" {
  description = "Name for the Glue Data Catalog database"
  type        = string
  default     = "hometech_catalog_db"
}

variable "processed_meter_table_name" {
  description = "Name for the Glue table storing processed meter data"
  type        = string
  default     = "processed_meter_data"
}

variable "glue_script_local_path" {
  description = "Local path to the Glue Python script file"
  type        = string
  default     = "../../scripts/glue_ingest_meter_data_v2.py" # Relative path example
}

variable "glue_script_s3_key" {
  description = "S3 key (path within bucket) for the Glue Python script"
  type        = string
  default     = "scripts/glue_ingest_meter_data_v2.py"
}

variable "glue_job_name" {
  description = "Name of the Glue ETL job"
  type        = string
  default     = "hometech-ingest-meter-data"
}
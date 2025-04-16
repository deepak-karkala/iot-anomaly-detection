# --- Outputs ---
output "raw_data_bucket_name" {
  value = aws_s3_bucket.raw_data.bucket
}

output "processed_data_bucket_name" {
  value = aws_s3_bucket.processed_data.bucket
}

output "glue_scripts_bucket_name" {
  value = aws_s3_bucket.glue_scripts.bucket
}

output "glue_job_name" {
  value = aws_glue_job.ingest_meter_data_job.name
}

output "glue_catalog_database_name" {
  value = aws_glue_catalog_database.hometech_db.name
}

output "glue_catalog_table_name" {
  value = aws_glue_catalog_table.processed_meter_data_table.name
}

output "glue_role_arn" {
  value = aws_iam_role.glue_service_role.arn
}
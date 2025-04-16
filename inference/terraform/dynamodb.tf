resource "aws_dynamodb_table" "alert_table" {
  name           = local.alert_db_table_name_full
  billing_mode   = "PAY_PER_REQUEST" # Or PROVISIONED if load is predictable
  hash_key       = "AlertID"        # Example: ApartmentID#YYYY-MM-DD

  attribute {
    name = "AlertID"
    type = "S" # String
  }

  # Optional: Add Sort Key if needed (e.g., Timestamp)
  # range_key = "AlertTimestamp"
  # attribute { name = "AlertTimestamp"; type = "S" } # String for ISO format

  # Optional: Define GSIs for querying (e.g., by ApartmentID and Status)
  # global_secondary_index {
  #   name            = "ApartmentStatusIndex"
  #   hash_key        = "ApartmentID"
  #   range_key       = "Status" # Query specific status per apartment
  #   projection_type = "INCLUDE" # Or ALL or KEYS_ONLY
  #   non_key_attributes = ["EventDate", "AnomalyScore"] # Attributes to project
  #   # Billing mode applies to GSI too if PAY_PER_REQUEST
  # }
  # attribute { name = "ApartmentID"; type = "S"}
  # attribute { name = "Status"; type = "S"}


  tags = local.tags

  # Optional: Enable TTL
  # ttl {
  #   attribute_name = "ExpiresAt" # Need to add this attribute when writing items
  #   enabled        = true
  # }
}
'''
Script 1: Schema Validation (scripts/validate_schema.py)
	- Purpose: Runs as a SageMaker Processing Job. Reads processed data, compares its
		schema to an expected definition, and fails if there's a mismatch.
	- Assumptions: Expected schema is provided, perhaps as a JSON file in S3 or passed
		as an argument string.
'''

import argparse
import json
import logging
import sys

from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --------------------

def load_expected_schema(schema_path):
    """Loads expected schema from a JSON string or file."""
    try:
        # Option 1: Assume schema_path is a JSON string argument
        # return json.loads(schema_path)
        # Option 2: Assume schema_path is an S3 path to a JSON file (more robust)
        # This requires Spark or boto3 to read from S3.
        # For simplicity here, let's assume it's passed as a JSON string for now.
        # In a real implementation, reading from S3 is better.
        logger.info(f"Parsing expected schema from argument string.")
        expected_schema_dict = json.loads(schema_path)
        # Basic validation of loaded schema structure (example)
        if not isinstance(expected_schema_dict, dict) or "columns" not in expected_schema_dict:
             raise ValueError("Expected schema JSON must be an object with a 'columns' key.")
        logger.info(f"Successfully parsed expected schema.")
        return expected_schema_dict
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse expected schema JSON: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading expected schema from '{schema_path}': {e}")
        raise

def validate_schema(spark, data_path, expected_schema_dict):
    """Reads data and validates its schema against the expected definition."""
    logger.info(f"Reading data from {data_path} for schema validation.")
    try:
        # Read only schema or a small sample to avoid loading all data
        df = spark.read.format("parquet").load(data_path)
        actual_schema = df.schema.jsonValue() # Get schema as JSON dict
        logger.info(f"Actual schema inferred: {json.dumps(actual_schema, indent=2)}")

    except AnalysisException as e:
        logger.error(f"Could not read data or infer schema from path: {data_path}. Error: {e}")
        return False # Indicate failure
    except Exception as e:
        logger.error(f"An unexpected error occurred during data read: {e}")
        return False

    # --- Schema Comparison Logic ---
    # Simple comparison: check field names and types (case-insensitive names)
    expected_fields = {f['name'].lower(): f['type'] for f in expected_schema_dict['columns']}
    actual_fields = {f['name'].lower(): f['type'] for f in actual_schema['fields']}

    missing_in_actual = set(expected_fields.keys()) - set(actual_fields.keys())
    extra_in_actual = set(actual_fields.keys()) - set(expected_fields.keys())
    type_mismatches = {}

    for field_name, expected_type in expected_fields.items():
        if field_name in actual_fields:
            actual_type = actual_fields[field_name]
            # Basic type comparison (might need refinement based on Spark vs JSON types)
            if isinstance(actual_type, dict) and 'elementType' in actual_type: # Handle array/map types if needed
                 pass # Add specific comparison logic for complex types
            elif str(actual_type).lower() != str(expected_type).lower():
                type_mismatches[field_name] = {'expected': expected_type, 'actual': actual_type}

    # --- Report Results ---
    is_valid = True
    if missing_in_actual:
        logger.error(f"Schema Validation Failed: Missing columns in actual data: {missing_in_actual}")
        is_valid = False
    if extra_in_actual:
        logger.warning(f"Schema Validation Warning: Extra columns found in actual data: {extra_in_actual}")
        # Decide if extra columns are acceptable or should cause failure
        # is_valid = False # Uncomment if extra columns are not allowed
    if type_mismatches:
        logger.error(f"Schema Validation Failed: Type mismatches found: {type_mismatches}")
        is_valid = False

    if is_valid:
        logger.info("Schema Validation Successful: Actual schema matches expected schema.")
    else:
        logger.error("Schema Validation Failed.")

    return is_valid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="S3 path to the processed data directory.")
    parser.add_argument("--expected-schema", type=str, required=True, help="JSON string or S3 path containing the expected schema definition.")
    # Add arguments for date range filtering if validation should be on a subset
    # parser.add_argument("--start-date", type=str, required=False)
    # parser.add_argument("--end-date", type=str, required=False)

    args = parser.parse_args()

    spark = SparkSession.builder.appName("SchemaValidation").getOrCreate()

    try:
        expected_schema = load_expected_schema(args.expected_schema)
        # Add date filtering logic here if needed based on args.start_date/end_date
        # data_path_filtered = filter_path_by_date(args.data_path, args.start_date, args.end_date)
        data_path_filtered = args.data_path # Placeholder

        validation_passed = validate_schema(spark, data_path_filtered, expected_schema)

        if not validation_passed:
            sys.exit(1) # Exit with error code if validation fails

        logger.info("Schema validation process completed successfully.")

    except Exception as e:
        logger.error(f"Unhandled exception during schema validation: {e}", exc_info=True)
        sys.exit(1)

    finally:
        spark.stop()

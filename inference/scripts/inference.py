'''
- Purpose: 
	This script is packaged within the training Docker container
	(or a separate inference container if needed) and executed by SageMaker Batch Transform.
	It loads the model and performs predictions/scoring.
- Assumptions:
	Model artifacts are in /opt/ml/model. Input data is provided (e.g., as CSV) and
	output should be written to /opt/ml/output.
- Notes: 
	- SageMaker Batch Transform typically calls model_fn once, then predict_fn repeatedly.
	- It doesn't use input_fn/output_fn in the same way as real-time endpoints by default,
	- but defining them is good practice for container structure.
	- The script needs to be executable and handle input/output via specified paths/conventions
	- if not using the SageMaker Python SDK's framework handlers within the container.
	- For simpler custom containers, you might just load model once, loop through input files, predict,
		and write output files.
	- This script assumes a structure compatible with SageMaker's built-in framework handling or
		Python SDK abstractions.
'''

# scripts/inference.py
# This script runs INSIDE the Batch Transform job/container

import argparse
import logging
import os

import joblib
import pandas as pd

# Import necessary ML libraries (must match those used in training/saving)
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import LocalOutlierFactor

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --------------------

MODEL_DIR = "/opt/ml/model"
# Define based on features output by feature_engineering_inference.py (in order)
# EXCLUDE identifier columns if model wasn't trained on them
# MUST match the feature_columns saved in model.joblib
# feature_columns = [...] # Loaded from artifacts below

def model_fn(model_dir):
    """Loads the model artifact from the specified directory."""
    logger.info(f"Loading model from {model_dir}")
    try:
        model_file = os.path.join(model_dir, "model.joblib")
        artifacts = joblib.load(model_file)
        logger.info("Model artifacts loaded successfully.")
        # Extract individual components if needed
        # scaler = artifacts['scaler']
        # lr_model = artifacts['linear_regression']
        # lof_model = artifacts['local_outlier_factor'] # Or parameters
        # feature_columns = artifacts['feature_columns']
        return artifacts # Return the whole dictionary or specific components
    except Exception as e:
        logger.error(f"Failed to load model from {model_dir}: {e}", exc_info=True)
        raise # Fail the job if model doesn't load


def input_fn(request_body, request_content_type):
    """Parses input data for prediction."""
    logger.info(f"Received request with content type: {request_content_type}")
    # Batch Transform typically sends data line by line or in batches
    # Handle common types like CSV
    if request_content_type == 'text/csv':
        try:
            # Assuming no header, columns ordered as expected
            # Need to know the feature columns from the loaded model later
            df = pd.read_csv(io.StringIO(request_body), header=None)
            logger.info(f"Parsed CSV input with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to parse CSV input: {e}", exc_info=True)
            raise ValueError("Failed to parse CSV input") from e
    # Add handling for other types like 'application/jsonlines' if needed
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data_df, model_artifacts):
    """Generates anomaly scores using the loaded model."""
    logger.info("Starting prediction/scoring...")
    try:
        # --- Extract model components ---
        scaler = model_artifacts['scaler']
        lr_model = model_artifacts['linear_regression']
        lof_model = model_artifacts['local_outlier_factor'] # Or parameters/fitted object
        feature_columns = model_artifacts['feature_columns'] # CRITICAL: Get expected columns

        # Assign column names if input_fn read header=None CSV
        if len(input_data_df.columns) == len(feature_columns):
             input_data_df.columns = feature_columns
        else:
             # If identifier columns were included in feature output, handle them
             # Example: Assume identifiers are first N columns
             num_ids = 3 # e.g., apartment_id, building_id, event_date
             all_cols = ["id_col1", "id_col2", "id_col3"] + feature_columns # Placeholder names
             if len(input_data_df.columns) == len(all_cols):
                  input_data_df.columns = all_cols
                  numeric_features_df = input_data_df[feature_columns]
             else:
                  raise ValueError(f"Input data columns ({len(input_data_df.columns)}) do not match expected feature columns ({len(feature_columns)})")
        logger.info("Assigned column names.")

        # --- Scoring Logic (Mirrors evaluation script but without labels) ---
        # Handle NaNs (should be handled in feature engineering ideally)
        numeric_features_df = numeric_features_df.fillna(0)

        # 1. Scale
        scaled_features = scaler.transform(numeric_features_df)
        logger.info("Applied scaling.")

        # 2. LR Residuals
        X_lr = numeric_features_df[['hdd']] # Match training
        lr_predictions = lr_model.predict(X_lr)
        lr_residuals = numeric_features_df['daily_energy_kwh'] - lr_predictions
        logger.info("Calculated LR residuals.")

        # 3. LOF Scores
        # This depends heavily on how LOF was saved/intended for use.
        # Option A: Use negative_outlier_factor_ from training (less ideal for new points)
        # Assuming lof_model here IS the fitted object:
        # Check if negative_outlier_factor_ can be used directly or if predict/score_samples is needed.
        # Sklearn LOF `predict` method needs `novelty=True` during init and fit on training data only.
        # If novelty=False (as in training script), `fit_predict` is common, but that includes the points themselves.
        # Safest might be to re-evaluate: What *exactly* should LOF contribute to the anomaly score for *new* data?
        # Placeholder: Using negative_outlier_factor_. **REVIEW THIS BASED ON AD STRATEGY**
        # If the number of input points != number of training points, this attribute won't match.
        # Re-fitting LOF here or using score_samples might be needed, which implies saving scaler too.
        # Let's assume for now we use the score_samples method if available, or just use LR.
        try:
            # Try using score_samples if the object supports it
            lof_scores = lof_model.score_samples(scaled_features) # Higher is better (more normal)
            logger.info("Calculated LOF scores using score_samples.")
        except AttributeError:
             # Fallback or alternative scoring if score_samples isn't applicable/saved
             logger.warning("LOF model object does not support score_samples or was not saved appropriately. Using placeholder LOF score.")
             lof_scores = np.zeros(len(input_data_df)) # Placeholder

        # --- Combine Scores ---
        # Example: Standardized LR Residual + (-LOF Score)
        # Make sure output is easily parseable (e.g., add scores as new columns)
        output_df = input_data_df.copy() # Start with input (potentially including IDs)
        output_df['anomaly_score_lr_residual'] = lr_residuals
        output_df['anomaly_score_lof'] = lof_scores # Higher = more normal
        # Combine into a final score if desired
        output_df['anomaly_score_combined'] = output_df['anomaly_score_lr_residual'].abs() - output_df['anomaly_score_lof'] # Example
        logger.info("Combined scores calculated.")

        # Select columns to output (e.g., IDs + scores)
        # Batch Transform writes the first column as prediction by default if no output_fn
        # Best practice: Use output_fn or ensure desired score is first / return structured data
        # Let's return IDs and the combined score for simplicity here
        result = output_df[['id_col1', 'id_col2', 'id_col3', 'anomaly_score_combined']] # Adjust ID col names

        return result # Return Pandas DataFrame

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        # Return structure indicating error? Batch transform might handle exceptions.
        # Return None or raise error
        raise


def output_fn(prediction_output, accept):
    """Serializes the prediction output."""
    logger.info(f"Serializing prediction output for accept type: {accept}")
    if accept == 'text/csv':
        # Convert DataFrame to CSV string
        return prediction_output.to_csv(header=False, index=False)
    elif accept == 'application/jsonlines':
         # Convert DataFrame to JSON Lines string
        return prediction_output.to_json(orient='records', lines=True)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")

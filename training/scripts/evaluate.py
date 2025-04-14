'''
Script 4: Model Evaluation (scripts/evaluate.py)
	- Purpose:
		Runs as a SageMaker Processing Job.
		Loads the trained model, runs it on evaluation data, calculates metrics,
			checks throughput, and outputs a JSON report.
	- Assumptions:
		Model artifact (model.joblib) available at /opt/ml/input/data/model.
		Evaluation features at /opt/ml/input/data/eval_features. Historical labels might be optional
'''

import argparse
import json
import logging
import os
import time

import joblib
import pandas as pd
# --- Import ML libraries needed for prediction/scoring ---
from sklearn.metrics import f1_score  # Example if using labels
from sklearn.metrics import precision_score, recall_score

# from sklearn.neighbors import LocalOutlierFactor # If needed for prediction

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --------------------

# Define paths based on SageMaker Processing Job environment
BASE_PATH = "/opt/ml/processing"
MODEL_PATH = os.path.join(BASE_PATH, "model")
EVAL_FEATURES_PATH = os.path.join(BASE_PATH, "input", "eval_features")
# HISTORICAL_LABELS_PATH = os.path.join(BASE_PATH, "input", "labels") # Optional
OUTPUT_PATH = os.path.join(BASE_PATH, "evaluation")


def calculate_metrics(model_artifacts, eval_df, historical_labels_df=None):
    """Calculates performance metrics for the trained model."""
    logger.info("Calculating evaluation metrics...")
    metrics = {}

    # --- Load Model Components ---
    try:
        scaler = model_artifacts['scaler']
        lr_model = model_artifacts['linear_regression']
        lof_model = model_artifacts['local_outlier_factor'] # Or load LOF parameters
        feature_columns = model_artifacts['feature_columns']
        logger.info("Model components loaded.")
    except KeyError as e:
        logger.error(f"Missing component in model artifact: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading model components: {e}")
        raise

    # --- Prepare Data ---
    if eval_df.empty:
        logger.warning("Evaluation DataFrame is empty. Cannot calculate metrics.")
        return {"status": "NoData", "message": "Evaluation data empty"}

    # Ensure all required feature columns are present
    missing_cols = [c for c in feature_columns if c not in eval_df.columns]
    if missing_cols:
         logger.error(f"Evaluation data missing required feature columns: {missing_cols}")
         raise ValueError("Missing feature columns in evaluation data.")

    # Handle missing values in eval data (should ideally use same strategy as training)
    eval_df = eval_df.fillna(0) # Placeholder
    numeric_features = eval_df[feature_columns]

    # --- Generate Scores/Predictions (Placeholder Logic) ---
    # This part is highly dependent on how you define/use anomalies from LR+LOF
    try:
        logger.info("Generating anomaly scores/predictions...")
        # 1. Scale features
        scaled_features = scaler.transform(numeric_features)

        # 2. Get LR residuals (as potential anomaly score component)
        X_lr = eval_df[['hdd']].fillna(0) # Match training feature
        lr_predictions = lr_model.predict(X_lr)
        lr_residuals = eval_df['daily_energy_kwh'] - lr_predictions
        # You might standardize residuals or use absolute values

        # 3. Get LOF scores
        # If LOF object saved directly and supports prediction (like sklearn's fit_predict)
        # lof_scores = lof_model.score_samples(scaled_features) # Returns opposite of outlier score
        # lof_predictions = lof_model.fit_predict(scaled_features) # -1 for outlier, 1 for inlier

        # Placeholder: Assume we derive a combined anomaly score
        # This needs to be defined based on how LR and LOF results are combined
        # Example: standardized residual + inverse LOF score?
        eval_df['anomaly_score_lr'] = lr_residuals # Example
        eval_df['anomaly_score_lof'] = lof_model.negative_outlier_factor_ # Uses fitted data - needs care!
        # A better approach for LOF scoring on *new* data might be needed depending on library/impl.
        # Placeholder combined score:
        eval_df['anomaly_score_combined'] = eval_df['anomaly_score_lr'].abs() - eval_df['anomaly_score_lof'] # Arbitrary example


        # --- Calculate Performance Metrics ---
        # Example: Distribution of scores
        metrics['score_mean'] = float(eval_df['anomaly_score_combined'].mean())
        metrics['score_stddev'] = float(eval_df['anomaly_score_combined'].std())
        metrics['score_min'] = float(eval_df['anomaly_score_combined'].min())
        metrics['score_max'] = float(eval_df['anomaly_score_combined'].max())
        metrics['num_records_evaluated'] = int(eval_df.shape[0])

        # Example: If using historical labels (e.g., from manual feedback)
        # Assume historical_labels_df has 'apartment_record_id', 'event_date', 'is_anomaly' (0 or 1)
        if historical_labels_df is not None and not historical_labels_df.empty:
            logger.info("Calculating metrics against historical labels.")
            eval_df_with_labels = pd.merge(
                eval_df, historical_labels_df,
                on=["apartment_record_id", "event_date"], # Adjust keys as needed
                how="left"
            )
            eval_df_with_labels['is_anomaly'] = eval_df_with_labels['is_anomaly'].fillna(0) # Assume unlabeled are not anomalies

            # Define a threshold on 'anomaly_score_combined' to make binary predictions
            anomaly_threshold = eval_df['anomaly_score_combined'].quantile(0.95) # Example: top 5% score
            eval_df_with_labels['predicted_anomaly'] = (eval_df_with_labels['anomaly_score_combined'] > anomaly_threshold).astype(int)

            metrics['precision'] = precision_score(eval_df_with_labels['is_anomaly'], eval_df_with_labels['predicted_anomaly'])
            metrics['recall'] = recall_score(eval_df_with_labels['is_anomaly'], eval_df_with_labels['predicted_anomaly'])
            metrics['f1_score'] = f1_score(eval_df_with_labels['is_anomaly'], eval_df_with_labels['predicted_anomaly'])
            metrics['threshold_used'] = anomaly_threshold
            logger.info(f"Label-based metrics: { {k: metrics[k] for k in ['precision', 'recall', 'f1_score']} }")
        else:
            logger.warning("No historical labels provided or labels file empty, skipping precision/recall calculation.")

    except Exception as e:
        logger.error(f"Error during prediction or metric calculation: {e}", exc_info=True)
        metrics['status'] = "Error"
        metrics['error_message'] = str(e)
        return metrics


    metrics['status'] = "Success"
    logger.info(f"Calculated metrics: {metrics}")
    return metrics

def check_throughput(start_time, end_time, num_records):
    """Estimates processing throughput."""
    if num_records == 0:
        return 0.0
    duration_seconds = end_time - start_time
    if duration_seconds == 0:
        return float('inf') # Avoid division by zero
    throughput = num_records / duration_seconds
    logger.info(f"Estimated throughput: {throughput:.2f} records/sec (processed {num_records} in {duration_seconds:.2f}s)")
    return throughput


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Inputs automatically mapped by SageMaker Processing Job
    # Outputs automatically mapped by SageMaker Processing Job

    # Add optional argument for historical labels path if needed
    parser.add_argument('--historical-labels-s3-uri', type=str, default=None)
    # Add args to receive training job metadata if needed for throughput analysis
    parser.add_argument('--training-duration-seconds', type=float, default=None)
    parser.add_argument('--training-record-count', type=int, default=None)


    args = parser.parse_args()
    logger.info(f"Received arguments: {args}")

    evaluation_report = {}
    start_time = time.time()

    try:
        # --- Load Model ---
        model_file = os.path.join(MODEL_PATH, "model.joblib")
        logger.info(f"Loading model from {model_file}")
        model_artifacts = joblib.load(model_file)

        # --- Load Evaluation Features ---
        logger.info(f"Loading evaluation features from {EVAL_FEATURES_PATH}")
        eval_files = [os.path.join(EVAL_FEATURES_PATH, f) for f in os.listdir(EVAL_FEATURES_PATH) if f.endswith('.parquet')]
        if not eval_files:
             raise FileNotFoundError(f"No parquet files found in evaluation features path: {EVAL_FEATURES_PATH}")
        eval_df = pd.concat((pd.read_parquet(f) for f in eval_files), ignore_index=True)
        logger.info(f"Loaded evaluation features data with shape: {eval_df.shape}")


        # --- Load Historical Labels (Optional) ---
        historical_labels_df = None
        if args.historical_labels_s3_uri:
            try:
                # This assumes the processing job role has access to this path
                logger.info(f"Loading historical labels from {args.historical_labels_s3_uri}")
                # Use appropriate library (pandas, spark) depending on label format/size
                historical_labels_df = pd.read_parquet(args.historical_labels_s3_uri) # Example
                logger.info(f"Loaded historical labels data with shape: {historical_labels_df.shape}")
            except Exception as label_e:
                logger.warning(f"Could not load historical labels: {label_e}. Proceeding without them.")


        # --- Calculate Metrics ---
        performance_metrics = calculate_metrics(model_artifacts, eval_df, historical_labels_df)
        evaluation_report.update(performance_metrics)

    except Exception as e:
        logger.error(f"Exception during evaluation setup or metric calculation: {e}", exc_info=True)
        evaluation_report["status"] = "Error"
        evaluation_report["error_message"] = str(e)


    # --- Calculate/Check Throughput ---
    # Option 1: Use metadata passed from training job
    if args.training_record_count is not None and args.training_duration_seconds is not None:
        throughput = check_throughput(0, args.training_duration_seconds, args.training_record_count)
        evaluation_report["training_throughput_records_per_sec"] = throughput
    else:
        # Option 2: Estimate based on evaluation run time (less accurate for training)
        # This only measures evaluation throughput, not training.
        eval_throughput = check_throughput(start_time, time.time(), evaluation_report.get('num_records_evaluated', 0))
        evaluation_report["evaluation_throughput_records_per_sec"] = eval_throughput
        logger.warning("Throughput calculated based on evaluation step duration, not training.")

    # --- Save Evaluation Report ---
    output_file_path = os.path.join(OUTPUT_PATH, 'evaluation_report.json')
    logger.info(f"Saving evaluation report to {output_file_path}")
    try:
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        with open(output_file_path, 'w') as f:
            json.dump(evaluation_report, f, indent=4)
        logger.info("Evaluation report saved.")
    except Exception as e:
        logger.error(f"Failed to save evaluation report: {e}", exc_info=True)
        # Decide if this failure should fail the whole job
        # sys.exit(1)

    # Exit with error if metric calculation failed
    if evaluation_report.get("status") == "Error":
        sys.exit(1)

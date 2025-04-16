'''
- Verify the predict_fn logic, assuming model_fn loads the artifacts correctly.
	Focus on score calculation based on mock inputs and artifacts
'''

import numpy as np
import pandas as pd
import pytest
# Assuming the script is saved as 'inference.py'
# Adjust the import path if necessary
from inference import predict_fn  # Import the function to test


# --- Mock Model Artifacts ---
# Recreate the structure expected by predict_fn, potentially using simpler mocks
class MockScalerPredict:
    def transform(self, df): return df.to_numpy() # Simple pass-through or basic scaling

class MockLRPredict:
    def predict(self, df): return np.array([d[0] * 0.5 for d in df.to_numpy()]) # Example: energy = 0.5 * hdd

class MockLOFPredict:
    # How LOF scores are generated for *new* data depends heavily on implementation.
    # If using score_samples:
    def score_samples(self, df_scaled): return np.array([(-1.0 - i * 0.1) for i in range(len(df_scaled))]) # Example decreasing score
    # If using negative_outlier_factor_ (less ideal, assumes predict data matches fit data size):
    # negative_outlier_factor_ = np.array([-1.1, -1.5, -1.2]) # Example

@pytest.fixture
def mock_inference_artifacts():
    return {
        'scaler': MockScalerPredict(),
        'linear_regression': MockLRPredict(),
        'local_outlier_factor': MockLOFPredict(), # Using score_samples mock
        'feature_columns': ['daily_energy_kwh', 'hdd', 'avg_temp_diff'] # MUST match input data cols below
    }

# --- Test Cases ---

def test_predict_fn_success(mock_inference_artifacts):
    """Test successful prediction and scoring."""
    # Input data ROWS match expected features for the model artifact
    # Include identifier columns as output by feature_engineering_inference
    input_data = {
        "id_col1": ["apt1", "apt2", "apt3"], # apartment_id
        "id_col2": ["bldgA", "bldgA", "bldgB"], # building_id
        "id_col3": ["2024-01-18", "2024-01-18", "2024-01-18"], # event_date
        "daily_energy_kwh": [10.0, 5.0, 12.0],
        "hdd": [8.0, 8.0, 10.0],
        "avg_temp_diff": [-0.5, -0.8, -0.2],
        # Add dummy columns if feature eng outputs more than model needs
        # "extra_feature": [1, 2, 3]
    }
    input_df = pd.DataFrame(input_data)
    # Reorder columns to match how feature eng might output before selection in predict_fn
    # input_df = input_df[["id_col1", "id_col2", "id_col3", "daily_energy_kwh", "hdd", "avg_temp_diff", "extra_feature"]]

    output_df = predict_fn(input_df, mock_inference_artifacts)

    # Assertions
    assert isinstance(output_df, pd.DataFrame)
    assert output_df.shape[0] == 3 # Number of input rows
    assert "anomaly_score_combined" in output_df.columns
    assert "id_col1" in output_df.columns # Check IDs are retained

    # Check calculated scores based on mock logic
    # LR predicts energy = 0.5 * hdd => [4.0, 4.0, 5.0]
    # Residual = Actual - Predicted => [10-4, 5-4, 12-5] = [6.0, 1.0, 7.0]
    # LOF score_samples => [-1.0, -1.1, -1.2] (higher = more normal)
    # Combined = abs(Residual) - LOF Score => [6 - (-1.0), 1 - (-1.1), 7 - (-1.2)] = [7.0, 2.1, 8.2]
    expected_scores = [7.0, 2.1, 8.2]
    assert output_df["anomaly_score_combined"].tolist() == pytest.approx(expected_scores)

def test_predict_fn_missing_feature_column(mock_inference_artifacts):
    """Test error when input data misses a feature column needed by the model."""
    input_data = {
        "id_col1": ["apt1"], "id_col2": ["bldgA"], "id_col3": ["2024-01-18"],
        "daily_energy_kwh": [10.0],
        # "hdd": [8.0], # Missing HDD required by mock LR model
        "avg_temp_diff": [-0.5],
    }
    input_df = pd.DataFrame(input_data)

    # Predict function tries to access 'hdd' inside the mock LR predict
    with pytest.raises(KeyError): # Or potentially different error depending on access pattern
         predict_fn(input_df, mock_inference_artifacts)


def test_predict_fn_empty_input(mock_inference_artifacts):
    """Test with empty input dataframe."""
    input_df = pd.DataFrame(columns=["id_col1", "id_col2", "id_col3"] + mock_inference_artifacts['feature_columns'])
    output_df = predict_fn(input_df, mock_inference_artifacts)
    assert output_df.empty
    assert "anomaly_score_combined" in output_df.columns # Check schema is still applied

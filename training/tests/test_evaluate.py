import json
import os

import numpy as np
import pandas as pd
import pytest
# Assuming the script is saved as 'evaluate.py'
# Adjust the import path if necessary
from evaluate import calculate_metrics, check_throughput  # Import functions


# Mock model artifacts dictionary structure (simplified)
@pytest.fixture
def mock_model_artifacts():
    # Mock scaler (doesn't need actual fitting for metric calculation test)
    class MockScaler:
        def transform(self, df): return df.to_numpy() # Just return numpy array
    # Mock LR (needs predict)
    class MockLR:
        def predict(self, df): return np.array([10] * len(df)) # Predict constant value
    # Mock LOF (needs negative_outlier_factor_ attribute)
    class MockLOF:
        # Simulate the attribute accessed in the current evaluate script
        negative_outlier_factor_ = np.array([-1.1, -1.0, -2.5, -1.05]) # Need len=4 for sample data
    return {
        'scaler': MockScaler(),
        'linear_regression': MockLR(),
        'local_outlier_factor': MockLOF(), # Or parameters if evaluation uses them differently
        'feature_columns': ['daily_energy_kwh', 'hdd', 'avg_temp_diff'] # Minimal set
    }

@pytest.fixture
def sample_eval_data():
    """Sample evaluation data."""
    data = {
        "apartment_record_id": ["apt1", "apt1", "apt2", "apt2"],
        "event_date": pd.to_datetime(["2024-01-18", "2024-01-19", "2024-01-17", "2024-01-18"]),
        "daily_energy_kwh": [9.0, 15.0, 7.0, 8.0], # Day 2 for Apt1 has higher energy
        "hdd": [9.0, 8.0, 11.0, 9.0],
        "avg_temp_diff": [-0.5, -0.2, -0.6, -0.5],
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_historical_labels():
    """Sample historical labels."""
    data = {
        "apartment_record_id": ["apt1", "apt2"],
        "event_date": pd.to_datetime(["2024-01-19", "2024-01-17"]), # Corresponds to high energy day & one normal day
        "is_anomaly": [1, 0] # Label high energy day as anomaly
    }
    return pd.DataFrame(data)

# --- Test Cases for calculate_metrics ---

def test_calculate_metrics_success_no_labels(mock_model_artifacts, sample_eval_data):
    """Test metrics calculation without historical labels."""
    metrics = calculate_metrics(mock_model_artifacts, sample_eval_data, None)

    assert metrics["status"] == "Success"
    assert "score_mean" in metrics
    assert "score_stddev" in metrics
    assert "score_min" in metrics
    assert "score_max" in metrics
    assert metrics["num_records_evaluated"] == 4
    # Check label-based metrics are NOT present
    assert "precision" not in metrics
    assert "recall" not in metrics
    assert "f1_score" not in metrics


def test_calculate_metrics_success_with_labels(mock_model_artifacts, sample_eval_data, sample_historical_labels):
    """Test metrics calculation including historical labels."""
    metrics = calculate_metrics(mock_model_artifacts, sample_eval_data, sample_historical_labels)

    assert metrics["status"] == "Success"
    assert "score_mean" in metrics
    assert metrics["num_records_evaluated"] == 4
    # Check label-based metrics ARE present
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "threshold_used" in metrics
    # Values depend heavily on the mock scoring logic and threshold (quantile)
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0

def test_calculate_metrics_empty_eval_data(mock_model_artifacts):
    """Test with empty evaluation dataframe."""
    empty_eval_df = pd.DataFrame(columns=mock_model_artifacts['feature_columns'])
    metrics = calculate_metrics(mock_model_artifacts, empty_eval_df, None)
    assert metrics["status"] == "NoData"
    assert "message" in metrics


def test_calculate_metrics_missing_feature_column(mock_model_artifacts, sample_eval_data):
    """Test when evaluation data is missing a required feature column."""
    eval_data_missing_col = sample_eval_data.drop(columns=['hdd'])
    with pytest.raises(ValueError, match="Missing feature columns in evaluation data."):
        calculate_metrics(mock_model_artifacts, eval_data_missing_col, None)


def test_calculate_metrics_missing_model_artifact(sample_eval_data):
    """Test when model artifact dict is missing a required component."""
    bad_artifacts = {'scaler': object(), 'feature_columns': ['colA']} # Missing lr/lof
    with pytest.raises(KeyError): # Or specific error depending on implementation
        calculate_metrics(bad_artifacts, sample_eval_data, None)


# --- Test Cases for check_throughput ---

def test_check_throughput_valid():
    """Test basic throughput calculation."""
    throughput = check_throughput(start_time=100.0, end_time=110.0, num_records=50)
    assert throughput == pytest.approx(5.0) # 50 records / 10 seconds


def test_check_throughput_zero_records():
    """Test throughput with zero records."""
    throughput = check_throughput(start_time=100.0, end_time=110.0, num_records=0)
    assert throughput == 0.0


def test_check_throughput_zero_duration():
    """Test throughput with zero duration (should return infinity)."""
    throughput = check_throughput(start_time=100.0, end_time=100.0, num_records=50)
    assert throughput == float('inf')

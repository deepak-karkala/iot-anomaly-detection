import os

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
# Assuming the script is saved as 'train.py' and classes are importable
# Adjust the import path if necessary
from train import (BaseADModel, LR_LOF_Model,  # Import necessary components
                   get_model_strategy)

# Define feature columns used in tests
TEST_FEATURE_COLS = [
    "daily_energy_kwh", "avg_temp_diff", "hdd", "avg_temp_c",
    "energy_lag_1d", "energy_roll_avg_7d"
]

@pytest.fixture
def sample_feature_data():
    """Pytest fixture providing sample Pandas DataFrame for training."""
    data = {
        "apartment_record_id": ["apt1", "apt1", "apt1", "apt2", "apt2"],
        "event_date": pd.to_datetime(["2024-01-15", "2024-01-16", "2024-01-17", "2024-01-15", "2024-01-16"]),
        "building_id": ["bldgA", "bldgA", "bldgA", "bldgA", "bldgA"],
        "daily_energy_kwh": [10.0, 12.0, 8.0, 5.0, 6.0],
        "avg_temp_diff": [-0.75, -0.3, -0.3, -0.75, -0.4],
        "hdd": [10.0, 12.0, 11.0, 10.0, 12.0],
        "avg_temp_c": [5.0, 3.0, 4.0, 5.0, 3.0],
        "energy_lag_1d": [None, 10.0, 12.0, None, 5.0], # Include None/NaNs
        "energy_roll_avg_7d": [10.0, 11.0, 10.0, 5.0, 5.5],
    }
    return pd.DataFrame(data)

@pytest.fixture
def lr_lof_hyperparams():
    """Hyperparameters for LR_LOF_Model tests."""
    return {
        'feature_columns': TEST_FEATURE_COLS,
        'lof_neighbors': 3, # Use small number for testing
        'lof_contamination': 'auto'
    }

def test_lr_lof_model_fit_success(sample_feature_data, lr_lof_hyperparams):
    """Tests successful fitting of the LR_LOF_Model."""
    model = LR_LOF_Model(hyperparameters=lr_lof_hyperparams)
    model.fit(sample_feature_data)

    # Assertions
    assert model.model_artifacts is not None
    assert 'scaler' in model.model_artifacts
    assert 'linear_regression' in model.model_artifacts
    # assert 'local_outlier_factor' in model.model_artifacts # Check if actual object is stored
    assert 'local_outlier_factor_params' in model.model_artifacts # Check params are stored
    assert 'feature_columns' in model.model_artifacts

    assert isinstance(model.model_artifacts['scaler'], StandardScaler)
    assert isinstance(model.model_artifacts['linear_regression'], LinearRegression)
    # assert isinstance(model.model_artifacts['local_outlier_factor'], LocalOutlierFactor)
    assert model.model_artifacts['feature_columns'] == TEST_FEATURE_COLS
    # Check if models appear fitted (e.g., scaler has means)
    assert hasattr(model.model_artifacts['scaler'], 'mean_')
    assert hasattr(model.model_artifacts['linear_regression'], 'coef_')


def test_lr_lof_model_save_load(sample_feature_data, lr_lof_hyperparams, tmpdir):
    """Tests saving and loading the model artifacts."""
    model_path = str(tmpdir.join("test_model"))
    os.makedirs(model_path, exist_ok=True) # Create temp dir for model

    # Fit the model
    model = LR_LOF_Model(hyperparameters=lr_lof_hyperparams)
    model.fit(sample_feature_data)

    # Save the model
    model.save(model_path)
    saved_file = os.path.join(model_path, "model.joblib")
    assert os.path.exists(saved_file)

    # Load the model artifacts (not using a class load method here, just joblib)
    loaded_artifacts = joblib.load(saved_file)

    # Assertions on loaded artifacts
    assert isinstance(loaded_artifacts, dict)
    assert 'scaler' in loaded_artifacts
    assert 'linear_regression' in loaded_artifacts
    assert loaded_artifacts['feature_columns'] == TEST_FEATURE_COLS


def test_lr_lof_model_fit_missing_features_in_hyperparams(sample_feature_data):
    """Tests ValueError when feature_columns missing in hyperparams."""
    bad_hyperparams = {'lof_neighbors': 3} # Missing feature_columns
    model = LR_LOF_Model(hyperparameters=bad_hyperparams)
    with pytest.raises(ValueError, match="Hyperparameter 'feature_columns' must be provided"):
        model.fit(sample_feature_data)


def test_lr_lof_model_fit_missing_features_in_data(sample_feature_data, lr_lof_hyperparams):
    """Tests KeyError when features listed in hyperparams are missing from data."""
    data_missing_col = sample_feature_data.drop(columns=['hdd']) # Drop hdd
    model = LR_LOF_Model(hyperparameters=lr_lof_hyperparams) # Expects 'hdd'
    with pytest.raises(KeyError): # Expect key error when accessing missing 'hdd'
        model.fit(data_missing_col)


def test_get_model_strategy_factory(lr_lof_hyperparams):
    """Tests the factory function."""
    model = get_model_strategy("LR_LOF", lr_lof_hyperparams)
    assert isinstance(model, LR_LOF_Model)
    assert model.hyperparameters == lr_lof_hyperparams

    with pytest.raises(ValueError, match="Unknown model strategy: UnknownModel"):
        get_model_strategy("UnknownModel", {})

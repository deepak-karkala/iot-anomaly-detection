'''
Script 3: Model Training (scripts/train/train.py)
	- Purpose:
		Runs inside a SageMaker Training Job container.
		Reads features, trains the AD model, saves the model artifact.
	- Assumptions:
		Feature data is available at /opt/ml/input/data/features.
		Hyperparameters are passed as arguments. Model components need to be saved to /opt/ml/model.
'''

import abc  # Abstract Base Class module
import argparse
import logging
import os

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
# --- Import your ML libraries ---
from sklearn.preprocessing import StandardScaler

# Add imports for other potential models (e.g., IsolationForest, Autoencoder libraries)
# import numpy as np # Example

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --------------------

# === Strategy Pattern Implementation ===

class BaseADModel(abc.ABC):
    """Abstract Base Class for Anomaly Detection models."""

    def __init__(self, hyperparameters=None):
        self.hyperparameters = hyperparameters if hyperparameters else {}
        self.model_artifacts = {} # Dictionary to store fitted components
        logger.info(f"Initialized {self.__class__.__name__} with params: {self.hyperparameters}")

    @abc.abstractmethod
    def fit(self, features_df):
        """Fits the model components to the training data."""
        pass

    @abc.abstractmethod
    def save(self, model_path):
        """Saves the fitted model artifacts."""
        pass

    # Optional: Add predict/score method if evaluation needs it directly
    # @abc.abstractmethod
    # def predict(self, features_df):
    #     """Generates anomaly scores or predictions."""
    #     pass

    # Optional: Add a load method if needed within training/evaluation
    # @classmethod
    # @abc.abstractmethod
    # def load(cls, model_path):
    #    """Loads the fitted model artifacts."""
    #    pass


class LR_LOF_Model(BaseADModel):
    """Concrete implementation using Linear Regression + Local Outlier Factor."""

    def fit(self, features_df):
        logger.info("Fitting LR_LOF_Model...")
        feature_columns = self.hyperparameters.get('feature_columns')
        if not feature_columns:
            raise ValueError("Hyperparameter 'feature_columns' must be provided for LR_LOF_Model")

        # --- Data Preparation ---
        features_df = features_df.fillna(0) # Simple imputation
        numeric_features = features_df[feature_columns]

        # 1. Fit Scaler
        logger.info("Fitting StandardScaler...")
        scaler = StandardScaler()
        scaler.fit(numeric_features)
        self.model_artifacts['scaler'] = scaler
        logger.info("StandardScaler fitted.")

        # 2. Fit Linear Regression (Example: Energy vs HDD)
        logger.info("Fitting Linear Regression (Energy ~ HDD)...")
        lr_model = LinearRegression()
        X_lr = features_df[['hdd']].fillna(0)
        y_lr = features_df['daily_energy_kwh']
        lr_model.fit(X_lr, y_lr)
        self.model_artifacts['linear_regression'] = lr_model
        logger.info("Linear Regression fitted.")

        # 3. Fit Local Outlier Factor
        lof_neighbors = self.hyperparameters.get('lof_neighbors', 20)
        lof_contamination = self.hyperparameters.get('lof_contamination', 'auto')
        logger.info(f"Fitting LOF with n_neighbors={lof_neighbors}, contamination={lof_contamination}...")
        lof_model = LocalOutlierFactor(n_neighbors=lof_neighbors, contamination=lof_contamination, novelty=False) # novelty=False for standard fit
        # Fit LOF on scaled data
        lof_model.fit(scaler.transform(numeric_features))
        # Store necessary attributes for potential later use (e.g., if needed for scoring)
        # Note: Storing the whole fitted object might be large or problematic depending on library version
        self.model_artifacts['local_outlier_factor_params'] = lof_model.get_params()
        # self.model_artifacts['local_outlier_factor_neg_factor'] = lof_model.negative_outlier_factor_ # Only if needed & available after fit
        self.model_artifacts['local_outlier_factor'] = lof_model # Store fitted object (use with caution)

        logger.info("LOF fitted.")
        self.model_artifacts['feature_columns'] = feature_columns # Store features used
        logger.info("LR_LOF_Model fitting complete.")


    def save(self, model_path):
        model_save_path = os.path.join(model_path, "model.joblib")
        logger.info(f"Saving LR_LOF_Model artifacts to {model_save_path}")
        if not self.model_artifacts:
             raise RuntimeError("Model has not been fitted yet. Call fit() before save().")
        try:
            joblib.dump(self.model_artifacts, model_save_path)
            logger.info("LR_LOF_Model artifacts saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save LR_LOF_Model artifacts: {e}", exc_info=True)
            raise

# --- Example of another potential Model Strategy ---
# class AutoencoderModel(BaseADModel):
#     def fit(self, features_df):
#         logger.info("Fitting AutoencoderModel...")
#         # --- Implementation using Keras/TensorFlow/PyTorch ---
#         # 1. Prepare data (scaling, maybe sequences)
#         # 2. Define Autoencoder architecture
#         # 3. Compile model
#         # 4. Train model
#         # 5. Store the trained model weights/architecture in self.model_artifacts
#         # Example: self.model_artifacts['keras_model'] = trained_keras_model
#         #          self.model_artifacts['scaler'] = scaler
#         #          self.model_artifacts['feature_columns'] = feature_columns
#         logger.info("AutoencoderModel fitting complete.")
#         pass # Replace with actual implementation

#     def save(self, model_path):
#         logger.info(f"Saving AutoencoderModel artifacts to {model_path}")
#         if not self.model_artifacts:
#              raise RuntimeError("Model has not been fitted yet. Call fit() before save().")
#         try:
#             # --- Implementation using appropriate library save functions ---
#             # Example Keras:
#             # self.model_artifacts['keras_model'].save(os.path.join(model_path, 'autoencoder_model.keras'))
#             # joblib.dump(self.model_artifacts['scaler'], os.path.join(model_path, 'scaler.joblib'))
#             # with open(os.path.join(model_path, 'features.json'), 'w') as f:
#             #     json.dump({'feature_columns': self.model_artifacts['feature_columns']}, f)
#             logger.info("AutoencoderModel artifacts saved successfully.")
#         except Exception as e:
#             logger.error(f"Failed to save AutoencoderModel artifacts: {e}", exc_info=True)
#             raise
#         pass # Replace with actual implementation


# === Factory Function (Optional but Clean) ===
def get_model_strategy(strategy_name, hyperparameters):
    """Factory function to return an instance of the requested model strategy."""
    if strategy_name == "LR_LOF":
        return LR_LOF_Model(hyperparameters=hyperparameters)
    # elif strategy_name == "Autoencoder":
    #     return AutoencoderModel(hyperparameters=hyperparameters)
    else:
        raise ValueError(f"Unknown model strategy: {strategy_name}")


# === Main Training Script Logic ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # --- SageMaker Framework Parameters ---
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/features'))
    parser.add_argument('--git-hash', type=str, default=None, help="Git commit hash of the training code")

    # --- Strategy Selection & Hyperparameters ---
    parser.add_argument('--model-strategy', type=str, required=True, choices=['LR_LOF'], help="Specify which model implementation to use.") # Add other choices like 'Autoencoder' later
    # Common Hyperparameters (can be unused by some strategies)
    parser.add_argument('--lof-neighbors', type=int, default=20)
    parser.add_argument('--lof-contamination', type=str, default='auto')
    # Add other strategy-specific hyperparameters here
    # parser.add_argument('--learning-rate', type=float, default=0.001) # Example for AE
    # parser.add_argument('--epochs', type=int, default=50) # Example for AE

    # Add argument for feature column names (could be read from config or passed)
    # Example: Pass as comma-separated string
    parser.add_argument('--feature-columns', type=str, required=True, help="Comma-separated list of feature column names")


    args = parser.parse_args()
    logger.info(f"Received arguments: {args}")

    # --- Load Data ---
    features_path = args.train
    logger.info(f"Loading features from {features_path}")
    try:
        all_files = [os.path.join(features_path, f) for f in os.listdir(features_path) if f.endswith('.parquet')]
        if not all_files: raise FileNotFoundError("No parquet files found.")
        features_df = pd.concat((pd.read_parquet(f) for f in all_files), ignore_index=True)
        logger.info(f"Loaded features data with shape: {features_df.shape}")
        if features_df.empty: raise ValueError("Feature DataFrame is empty.")
    except Exception as e:
        logger.error(f"Failed to load feature files: {e}", exc_info=True)
        sys.exit(1)


    # --- Instantiate and Train Selected Model Strategy ---
    try:
        # Parse feature columns string
        feature_col_list = [col.strip() for col in args.feature_columns.split(',')]

        # Prepare hyperparameters dict
        hyperparameters = vars(args) # Gets all args as a dict
        hyperparameters['feature_columns'] = feature_col_list # Add parsed features list
        # Remove non-hyperparameter args if needed
        del hyperparameters['model_dir']
        del hyperparameters['train']
        # del hyperparameters['model_strategy'] # Keep strategy if model needs it internally


        # Use factory or direct instantiation
        model_strategy = get_model_strategy(args.model_strategy, hyperparameters)
        # Or: if args.model_strategy == "LR_LOF": model_strategy = LR_LOF_Model(...)

        logger.info(f"Starting training for strategy: {args.model_strategy}")
        model_strategy.fit(features_df)
        logger.info(f"Completed training for strategy: {args.model_strategy}")

    except Exception as e:
         logger.error(f"Exception during model instantiation or fitting: {e}", exc_info=True)
         sys.exit(1)

    # --- Save Model Artifacts using Strategy's save method ---
    try:
        model_strategy.save(args.model_dir) # Pass the designated SageMaker model directory
    except Exception as e:
        logger.error(f"Exception during model saving: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Training script finished successfully.")

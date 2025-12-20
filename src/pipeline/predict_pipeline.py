# src/pipeline/predict_pipeline.py
import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


@dataclass
class PredictPipelineConfig:
    """Paths to model and preprocessor artifacts."""

    model_path: str = os.path.join("artifacts", "model.pkl")
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


class PredictPipeline:
    """
    Handles loading artifacts and making predictions.
    Used by Flask app and any batch prediction scripts.
    """

    def __init__(self):
        self.config = PredictPipelineConfig()

    def predict(self, features: pd.DataFrame):
        """
        Predict fraud for given input features.

        Args:
            features (pd.DataFrame): Raw input features (same schema as training)

        Returns:
            np.ndarray: Predicted labels (0/1)
        """
        try:
            logging.info("Starting prediction pipeline")

            # Load artifacts
            logging.info(f"Loading model from: {self.config.model_path}")
            model = load_object(self.config.model_path)

            logging.info(f"Loading preprocessor from: {self.config.preprocessor_path}")
            preprocessor = load_object(self.config.preprocessor_path)

            # Apply same feature engineering as training
            from src.components.data_transformation import FeatureEngineer

            feature_engineer = FeatureEngineer()

            logging.info("Applying feature engineering to input data")
            data_fe = feature_engineer.transform(features)

            # Separate features (drop target if present)
            if "isFraud" in data_fe.columns:
                data_fe = data_fe.drop(columns=["isFraud"])

            logging.info("Applying preprocessing pipeline")
            data_transformed = preprocessor.transform(data_fe)

            logging.info("Making predictions")
            preds = model.predict(data_transformed)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Container for a single transaction's raw inputs.
    Converts them into a DataFrame compatible with the pipeline.
    """

    def __init__(
        self,
        step: int,
        type: str,
        amount: float,
        nameOrig: str,
        oldbalanceOrg: float,
        newbalanceOrig: float,
        nameDest: str,
        oldbalanceDest: float,
        newbalanceDest: float,
        isFlaggedFraud: int = 0,
    ):
        self.step = step
        self.type = type
        self.amount = amount
        self.nameOrig = nameOrig
        self.oldbalanceOrg = oldbalanceOrg
        self.newbalanceOrig = newbalanceOrig
        self.nameDest = nameDest
        self.oldbalanceDest = oldbalanceDest
        self.newbalanceDest = newbalanceDest
        self.isFlaggedFraud = isFlaggedFraud

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the transaction into a single-row DataFrame.

        Returns:
            pd.DataFrame: One-row DataFrame with correct column names
        """
        try:
            data_dict = {
                "step": [self.step],
                "type": [self.type],
                "amount": [self.amount],
                "nameOrig": [self.nameOrig],
                "oldbalanceOrg": [self.oldbalanceOrg],
                "newbalanceOrig": [self.newbalanceOrig],
                "nameDest": [self.nameDest],
                "oldbalanceDest": [self.oldbalanceDest],
                "newbalanceDest": [self.newbalanceDest],
                "isFlaggedFraud": [self.isFlaggedFraud],
            }

            df = pd.DataFrame(data_dict)
            return df

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Quick manual test (will work after model is trained)
    try:
        sample = CustomData(
            step=1,
            type="TRANSFER",
            amount=50000.0,
            nameOrig="C123456789",
            oldbalanceOrg=60000.0,
            newbalanceOrig=10000.0,
            nameDest="C987654321",
            oldbalanceDest=0.0,
            newbalanceDest=50000.0,
            isFlaggedFraud=0,
        )

        sample_df = sample.to_dataframe()

        pipeline = PredictPipeline()
        pred = pipeline.predict(sample_df)

        print("\nSample prediction:", int(pred[0]))
        print("1 = Fraud, 0 = Legitimate\n")
    except Exception as e:
        print("Prediction test failed (likely because model isn't trained yet):", e)

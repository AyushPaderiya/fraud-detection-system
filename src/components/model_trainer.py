# src/components/model_trainer.py
import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    """Configuration for model training artifacts."""

    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """
    Trains multiple fraud detection models and selects the best performer.
    Uses PR-AUC as primary metric for imbalanced data.
    Memory-optimized: no cross-validation, no RandomizedSearchCV.
    """

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_single_model(self, model, X_train, y_train, X_val, y_val, model_name):
        """
        Train and evaluate a single model.

        Args:
            model: Sklearn-compatible model
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_name: Name for logging

        Returns:
            dict: Metrics for the model
        """
        try:
            logging.info(f"Training {model_name}...")

            model.fit(X_train, y_train)

            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)

            roc_auc = roc_auc_score(y_val, y_pred_proba)
            pr_auc = average_precision_score(y_val, y_pred_proba)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)

            logging.info(
                f"{model_name} - PR-AUC: {pr_auc:.4f}, " f"ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}"
            )

            return {
                "model": model,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, val_array):
        """
        Train multiple models and return the best one.

        Args:
            train_array (np.ndarray): Training data with target in last column
            val_array (np.ndarray): Validation data with target in last column

        Returns:
            tuple: (best_model_name, best_model_score, best_model_object)
        """
        try:
            logging.info("Starting model training")

            # Split features and target
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_val = val_array[:, :-1]
            y_val = val_array[:, -1]

            # Use at most ~300k rows for LR (tunable)
            max_lr_samples = 300_000

            if len(y_train) > max_lr_samples:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=max_lr_samples, random_state=42)
                ((lr_idx, _),) = sss.split(X_train, y_train)
                X_train_lr = X_train[lr_idx]
                y_train_lr = y_train[lr_idx]
            else:
                X_train_lr = X_train
                y_train_lr = y_train

            logging.info(f"Training data shape: {X_train.shape}")
            logging.info(f"Validation data shape: {X_val.shape}")

            # Class imbalance stats
            fraud_count = np.sum(y_train == 1)
            legit_count = np.sum(y_train == 0)
            scale_pos_weight = legit_count / fraud_count

            logging.info(
                f"Class distribution - Fraud: {fraud_count}, " f"Legitimate: {legit_count}"
            )
            logging.info(f"Scale pos weight: {scale_pos_weight:.2f}")

            # === Define Models (no hyperparameter tuning) ===
            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=500,  # avoid convergence issues
                    class_weight="balanced",
                    random_state=42,
                    solver="saga",  # good for large, sparse data
                    n_jobs=-1,
                    tol=1e-3,  # looser tolerance
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                    max_features="sqrt",
                ),
                "XGBoost": XGBClassifier(
                    n_estimators=100,
                    max_depth=7,
                    learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",  # faster on large data
                    eval_metric="aucpr",
                ),
                "LightGBM": LGBMClassifier(
                    n_estimators=100,
                    max_depth=7,
                    learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                ),
            }

            # === Train and evaluate all models ===
            logging.info("Training and evaluating models (no CV, no tuning)")
            model_report = {}

            for name, model in models.items():
                if name == "Logistic Regression":
                    metrics = self.evaluate_single_model(
                        model,
                        X_train_lr,
                        y_train_lr,  # use sampled data
                        X_val,
                        y_val,
                        name,
                    )
                else:
                    metrics = self.evaluate_single_model(
                        model,
                        X_train,
                        y_train,  # use full data
                        X_val,
                        y_val,
                        name,
                    )
                model_report[name] = metrics

            # === Select best model by PR-AUC ===
            logging.info("\n" + "=" * 70)
            logging.info("MODEL COMPARISON (PR-AUC)")
            logging.info("=" * 70)

            sorted_models = sorted(
                model_report.items(),
                key=lambda x: x[1]["pr_auc"],
                reverse=True,
            )

            for name, metrics in sorted_models:
                logging.info(
                    f"{name:20s} - PR-AUC: {metrics['pr_auc']:.4f}, "
                    f"ROC-AUC: {metrics['roc_auc']:.4f}, "
                    f"F1: {metrics['f1_score']:.4f}"
                )

            best_model_name = sorted_models[0][0]
            best_model_metrics = sorted_models[0][1]
            best_model_score = best_model_metrics["pr_auc"]
            best_model_object = best_model_metrics["model"]

            logging.info("=" * 70)
            logging.info(f"BEST MODEL: {best_model_name} " f"(PR-AUC: {best_model_score:.4f})")
            logging.info("=" * 70 + "\n")

            # Basic sanity check
            if best_model_score < 0.6:
                raise CustomException(
                    f"No model achieved acceptable performance. "
                    f"Best PR-AUC: {best_model_score:.4f}",
                    sys,
                )

            # === Save best model ===
            logging.info(
                f"Saving best model to: " f"{self.model_trainer_config.trained_model_file_path}"
            )
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model_object,
            )

            logging.info("Model training completed successfully")

            return best_model_name, best_model_score, best_model_object

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Optional: quick standalone test
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    print("\n" + "=" * 70)
    print("TESTING MODEL TRAINER")
    print("=" * 70 + "\n")

    ingestion = DataIngestion()
    train_path, val_path, test_path = ingestion.initiate_data_ingestion()
    print("✅ Data ingestion completed\n")

    transformation = DataTransformation()
    train_arr, val_arr, test_arr, _ = transformation.initiate_data_transformation(
        train_path, val_path, test_path
    )
    print("✅ Data transformation completed\n")

    trainer = ModelTrainer()
    best_name, best_score, best_model = trainer.initiate_model_trainer(train_arr, val_arr)

    print("\n" + "=" * 70)
    print("MODEL TRAINING TEST COMPLETED")
    print("=" * 70)
    print(f"Best Model: {best_name}")
    print(f"PR-AUC Score: {best_score:.4f}")
    print(f"Model saved at: artifacts/model.pkl")
    print("=" * 70 + "\n")

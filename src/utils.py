import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import RandomizedSearchCV
import joblib

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """
    Save Python object to pickle file.

    Args:
        file_path (str): Path where to save the object
        obj: Object to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Object saved successfully at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load Python object from pickle file.

    Args:
        file_path (str): Path to the pickle file

    Returns:
        Loaded object
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)

        logging.info(f"Object loaded successfully from: {file_path}")
        return obj

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params=None):
    """
    Train and evaluate multiple models with optional hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        models (dict): Dictionary of model name and model object
        params (dict): Dictionary of model name and hyperparameter grid

    Returns:
        dict: Model performance report with PR-AUC scores
    """
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training {model_name}...")

            # Hyperparameter tuning if params provided
            if params and model_name in params:
                logging.info(f"Performing hyperparameter tuning for {model_name}")

                rs = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=params[model_name],
                    n_iter=10,
                    cv=3,
                    verbose=1,
                    n_jobs=-1,
                    scoring="average_precision",  # PR-AUC for imbalanced data
                    random_state=42,
                )
                rs.fit(X_train, y_train)
                model = rs.best_estimator_

                logging.info(f"Best parameters for {model_name}: {rs.best_params_}")
            else:
                # Train with default parameters
                model.fit(X_train, y_train)

            # Make predictions
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            # Calculate metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            report[model_name] = {
                "model": model,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }

            logging.info(f"{model_name} - PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)


def calculate_business_metrics(y_true, y_pred, avg_fraud_amount=5000, investigation_cost=50):
    """
    Calculate business impact metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        avg_fraud_amount: Average monetary value of fraud
        investigation_cost: Cost to investigate each flagged transaction

    Returns:
        dict: Business metrics
    """
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        fraud_prevented = tp * avg_fraud_amount
        fraud_lost = fn * avg_fraud_amount
        investigation_costs = (tp + fp) * investigation_cost
        net_savings = fraud_prevented - investigation_costs

        total_fraud = (tp + fn) * avg_fraud_amount
        roi = (net_savings / total_fraud * 100) if total_fraud > 0 else 0

        metrics = {
            "fraud_prevented": fraud_prevented,
            "fraud_lost": fraud_lost,
            "investigation_costs": investigation_costs,
            "net_savings": net_savings,
            "roi": roi,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }

        return metrics

    except Exception as e:
        raise CustomException(e, sys)

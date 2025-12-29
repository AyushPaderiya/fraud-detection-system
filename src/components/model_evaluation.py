import os
import sys
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, calculate_business_metrics


from src.utils import read_yaml_config

@dataclass
class ModelEvaluationConfig:
    """Configuration for model evaluation artifacts."""

    config: dict = field(default_factory=lambda: read_yaml_config("config.yaml"))

    def __post_init__(self):
        artifacts_config = self.config['artifacts']['model_trainer'] # reuse trainer paths for report
        # Note: I'll add specific evaluation paths to config.yaml in a moment if needed, 
        # but let's stick to what's there or reasonable defaults based on config structure.
        self.evaluation_report_path = artifacts_config['evaluation_report_path']
        self.confusion_matrix_path = os.path.join("artifacts", "confusion_matrix.png")
        self.roc_curve_path = os.path.join("artifacts", "roc_curve.png")
        self.pr_curve_path = os.path.join("artifacts", "pr_curve.png")


def _to_python(obj):
    """Recursively convert numpy/pandas types to native Python types for JSON."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


class ModelEvaluation:
    """
    Comprehensive model evaluation for fraud detection.
    Generates metrics, visualizations, and business impact analysis.
    """

    def __init__(self):
        self.evaluation_config = ModelEvaluationConfig()

    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluate model performance with comprehensive metrics.

        Args:
            model: Trained model object
            X_test: Test features
            y_test: Test labels
            model_name: Name for logging

        Returns:
            tuple: (results_dict, y_pred, y_pred_proba)
        """
        try:
            logging.info(f"Evaluating {model_name} on test set")

            # Probabilities
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Optimal threshold by F1
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            optimal_idx = np.argmax(f1_scores[:-1])  # exclude last element
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

            # Apply threshold
            y_pred = (y_pred_proba >= optimal_threshold).astype(int)

            # Metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            # Precision at 90% recall
            idx_90 = np.argmin(np.abs(recalls - 0.90))
            precision_at_90_recall = float(precisions[idx_90])

            # Business metrics
            business_metrics = calculate_business_metrics(y_test, y_pred)

            results = {
                "model_name": model_name,
                "optimal_threshold": float(optimal_threshold),
                "metrics": {
                    "roc_auc": float(roc_auc),
                    "pr_auc": float(pr_auc),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "specificity": float(specificity),
                    "precision_at_90_recall": float(precision_at_90_recall),
                },
                "confusion_matrix": {
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_positives": int(tp),
                },
                "business_impact": business_metrics,
            }

            logging.info("\n" + "=" * 70)
            logging.info(f"EVALUATION RESULTS: {model_name}")
            logging.info("=" * 70)
            logging.info(f"ROC-AUC: {roc_auc:.4f}")
            logging.info(f"PR-AUC: {pr_auc:.4f}")
            logging.info(f"Optimal Threshold: {optimal_threshold:.4f}")
            logging.info(f"Precision: {precision:.4f}")
            logging.info(f"Recall: {recall:.4f}")
            logging.info(f"F1-Score: {f1:.4f}")
            logging.info(f"Precision @ 90% Recall: {precision_at_90_recall:.4f}")
            logging.info("Confusion Matrix:")
            logging.info(f"  TN: {tn:,} | FP: {fp:,}")
            logging.info(f"  FN: {fn:,} | TP: {tp:,}")
            logging.info("Business Impact:")
            logging.info(f"  Fraud Prevented: ${business_metrics['fraud_prevented']:,.0f}")
            logging.info(f"  Fraud Missed: ${business_metrics['fraud_lost']:,.0f}")
            logging.info(f"  Net Savings: ${business_metrics['net_savings']:,.0f}")
            logging.info(f"  ROI: {business_metrics['roi']:.2f}%")
            logging.info("=" * 70 + "\n")

            return results, y_pred, y_pred_proba

        except Exception as e:
            raise CustomException(e, sys)

    def plot_confusion_matrix(self, y_test, y_pred, save_path=None):
        """Generate and save confusion matrix visualization."""
        try:
            cm = confusion_matrix(y_test, y_pred)

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Legitimate", "Fraud"],
                yticklabels=["Legitimate", "Fraud"],
                cbar_kws={"label": "Count"},
            )
            plt.title("Confusion Matrix", fontsize=14, fontweight="bold", pad=20)
            plt.ylabel("True Label", fontsize=12)
            plt.xlabel("Predicted Label", fontsize=12)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logging.info(f"Confusion matrix saved to: {save_path}")

            plt.close()

        except Exception as e:
            raise CustomException(e, sys)

    def plot_roc_curve(self, y_test, y_pred_proba, save_path=None):
        """Generate and save ROC curve."""
        try:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color="#2E86AB", lw=3, label=f"Model (AUC = {roc_auc:.4f})")
            plt.plot([0, 1], [0, 1], color="#A23B72", lw=2, linestyle="--", label="Random Baseline")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate", fontsize=12, fontweight="bold")
            plt.ylabel("True Positive Rate (Recall)", fontsize=12, fontweight="bold")
            plt.title("ROC Curve", fontsize=14, fontweight="bold", pad=15)
            plt.legend(loc="lower right", fontsize=11)
            plt.grid(alpha=0.3, linestyle="--")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logging.info(f"ROC curve saved to: {save_path}")

            plt.close()

        except Exception as e:
            raise CustomException(e, sys)

    def plot_pr_curve(self, y_test, y_pred_proba, save_path=None):
        """Generate and save Precision-Recall curve."""
        try:
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)
            baseline = np.mean(y_test)

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color="#2E86AB", lw=3, label=f"Model (AUC = {pr_auc:.4f})")
            plt.axhline(
                y=baseline,
                color="#A23B72",
                linestyle="--",
                lw=2,
                label=f"Baseline ({baseline*100:.3f}% fraud)",
            )
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("Recall", fontsize=12, fontweight="bold")
            plt.ylabel("Precision", fontsize=12, fontweight="bold")
            plt.title("Precision-Recall Curve", fontsize=14, fontweight="bold", pad=15)
            plt.legend(loc="upper right", fontsize=11)
            plt.grid(alpha=0.3, linestyle="--")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logging.info(f"PR curve saved to: {save_path}")

            plt.close()

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self, model_path, preprocessor_path, test_path):
        """
        Complete model evaluation pipeline.

        Args:
            model_path: Path to trained model
            preprocessor_path: Path to fitted preprocessor
            test_path: Path to test CSV

        Returns:
            dict: Evaluation results (Python types)
        """
        try:
            logging.info("Starting model evaluation")

            # Load artifacts
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            test_df = pd.read_csv(test_path)

            # Feature engineering
            from src.components.data_transformation import FeatureEngineer

            feature_engineer = FeatureEngineer()
            test_df = feature_engineer.transform(test_df)

            # Separate features and target
            X_test = test_df.drop(columns=["isFraud"])
            y_test = test_df["isFraud"]

            # Transform features
            X_test_transformed = preprocessor.transform(X_test)

            # Evaluate
            results, y_pred, y_pred_proba = self.evaluate_model(model, X_test_transformed, y_test)

            # Plots
            self.plot_confusion_matrix(y_test, y_pred, self.evaluation_config.confusion_matrix_path)
            self.plot_roc_curve(y_test, y_pred_proba, self.evaluation_config.roc_curve_path)
            self.plot_pr_curve(y_test, y_pred_proba, self.evaluation_config.pr_curve_path)

            # Convert to JSON‑safe types
            results_py = _to_python(results)

            # Save JSON
            os.makedirs(
                os.path.dirname(self.evaluation_config.evaluation_report_path), exist_ok=True
            )
            with open(self.evaluation_config.evaluation_report_path, "w", encoding="utf-8") as f:
                json.dump(results_py, f, indent=4)
            logging.info(
                f"Evaluation report saved to: {self.evaluation_config.evaluation_report_path}"
            )

            logging.info("Model evaluation completed successfully")
            return results_py

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    evaluator = ModelEvaluation()
    results = evaluator.initiate_model_evaluation(
        model_path="artifacts/model.pkl",
        preprocessor_path="artifacts/preprocessor.pkl",
        test_path="artifacts/data/test.csv",
    )

    print("\n" + "=" * 70)
    print("MODEL EVALUATION COMPLETED")
    print("=" * 70)
    print(f"PR-AUC: {results['metrics']['pr_auc']:.4f}")
    print(f"ROC-AUC: {results['metrics']['roc_auc']:.4f}")
    print(f"F1-Score: {results['metrics']['f1_score']:.4f}")
    print(f"Net Savings: ${results['business_impact']['net_savings']:,.0f}")
    print("=" * 70 + "\n")

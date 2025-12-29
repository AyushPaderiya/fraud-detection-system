import os
import sys
import pandas as pd
from dataclasses import dataclass, field

from src.exception import CustomException
from src.logger import logging

from src.utils import read_yaml_config

@dataclass
class DataValidationConfig:
    """Configuration for data validation."""

    config: dict = field(default_factory=lambda: read_yaml_config("config.yaml"))

    def __post_init__(self):
        # We can add validation paths to config if we want, or use a default.
        # Let's add it to config.yaml if it's not there.
        self.validation_report_path = os.path.join("artifacts", "validation_report.txt")


class DataValidation:
    """
    Validates data quality and schema for fraud detection.
    Checks for missing values, duplicates, data types, and fraud distribution.
    """

    def __init__(self):
        self.validation_config = DataValidationConfig()

        # Expected schema for PaySim dataset
        self.expected_columns = [
            "step",
            "type",
            "amount",
            "nameOrig",
            "oldbalanceOrg",
            "newbalanceOrig",
            "nameDest",
            "oldbalanceDest",
            "newbalanceDest",
            "isFraud",
            "isFlaggedFraud",
        ]

        self.expected_types = {
            "step": "int64",
            "type": "object",
            "amount": "float64",
            "nameOrig": "object",
            "oldbalanceOrg": "float64",
            "newbalanceOrig": "float64",
            "nameDest": "object",
            "oldbalanceDest": "float64",
            "newbalanceDest": "float64",
            "isFraud": "int64",
            "isFlaggedFraud": "int64",
        }

    def validate_dataset(self, df, dataset_name="Dataset"):
        """
        Validate a single dataset.

        Args:
            df (pd.DataFrame): Dataset to validate
            dataset_name (str): Name for logging

        Returns:
            tuple: (is_valid, validation_report)
        """
        report = []
        is_valid = True

        report.append(f"\n{'='*70}")
        report.append(f"VALIDATION REPORT: {dataset_name}")
        report.append(f"{'='*70}\n")

        # Check 1: Column names
        report.append("1. Column Validation:")
        missing_cols = set(self.expected_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(self.expected_columns)

        if missing_cols:
            report.append(f"   ❌ Missing columns: {missing_cols}")
            is_valid = False
        else:
            report.append("   ✅ All expected columns present")

        if extra_cols:
            report.append(f"   ⚠️  Extra columns: {extra_cols}")

        # Check 2: Data types
        report.append("\n2. Data Type Validation:")
        type_mismatches = []
        for col, expected_type in self.expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type:
                    type_mismatches.append(f"{col}: expected {expected_type}, got {actual_type}")

        if type_mismatches:
            report.append(f"   ⚠️  Type mismatches:\n      " + "\n      ".join(type_mismatches))
        else:
            report.append("   ✅ All data types correct")

        # Check 3: Missing values
        report.append("\n3. Missing Values:")
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            report.append(f"   ⚠️  Total missing values: {missing_count}")
            missing_per_col = df.isnull().sum()[df.isnull().sum() > 0]
            for col, count in missing_per_col.items():
                report.append(f"      - {col}: {count} ({count/len(df)*100:.2f}%)")
        else:
            report.append("   ✅ No missing values")

        # Check 4: Duplicates
        report.append("\n4. Duplicate Records:")
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            report.append(f"   ⚠️  Duplicate rows: {dup_count} ({dup_count/len(df)*100:.2f}%)")
        else:
            report.append("   ✅ No duplicates")

        # Check 5: Target variable distribution
        report.append("\n5. Target Variable (isFraud):")
        if "isFraud" in df.columns:
            fraud_count = df["isFraud"].sum()
            fraud_rate = fraud_count / len(df) * 100
            report.append(f"   Fraud cases: {fraud_count} ({fraud_rate:.4f}%)")
            report.append(f"   Legitimate cases: {len(df) - fraud_count} ({100-fraud_rate:.4f}%)")

            if fraud_rate < 0.01:
                report.append("   ⚠️  Very low fraud rate (< 0.01%)")
            elif fraud_rate > 10:
                report.append("   ⚠️  Unusually high fraud rate (> 10%)")
            else:
                report.append("   ✅ Fraud rate within expected range")

        # Check 6: Transaction types
        report.append("\n6. Transaction Types:")
        if "type" in df.columns:
            type_dist = df["type"].value_counts()
            for txn_type, count in type_dist.items():
                report.append(f"   - {txn_type}: {count} ({count/len(df)*100:.2f}%)")

        # Check 7: Amount distribution
        report.append("\n7. Amount Statistics:")
        if "amount" in df.columns:
            report.append(f"   Min: ${df['amount'].min():.2f}")
            report.append(f"   Max: ${df['amount'].max():.2f}")
            report.append(f"   Mean: ${df['amount'].mean():.2f}")
            report.append(f"   Median: ${df['amount'].median():.2f}")

            if (df["amount"] < 0).any():
                report.append("   ❌ Negative amounts found!")
                is_valid = False

        report.append(f"\n{'='*70}")
        report.append(f"Overall Status: {'✅ PASSED' if is_valid else '❌ FAILED'}")
        report.append(f"{'='*70}\n")

        return is_valid, "\n".join(report)

    def initiate_data_validation(self, train_path, val_path, test_path):
        """
        Validate all datasets (train, val, test).

        Args:
            train_path (str): Path to training CSV
            val_path (str): Path to validation CSV
            test_path (str): Path to test CSV

        Returns:
            bool: True if all validations pass
        """
        logging.info("Starting data validation")

        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)

            # Validate each dataset
            train_valid, train_report = self.validate_dataset(train_df, "Training Set")
            val_valid, val_report = self.validate_dataset(val_df, "Validation Set")
            test_valid, test_report = self.validate_dataset(test_df, "Test Set")

            # Combine reports
            full_report = train_report + "\n" + val_report + "\n" + test_report

            # Save report
            os.makedirs(
                os.path.dirname(self.validation_config.validation_report_path), exist_ok=True
            )
            with open(self.validation_config.validation_report_path, "w", encoding="utf-8") as f:
                f.write(full_report)

            logging.info(
                f"Validation report saved to: {self.validation_config.validation_report_path}"
            )

            # Print to console
            print(full_report)

            # Overall result
            all_valid = train_valid and val_valid and test_valid

            if all_valid:
                logging.info("✅ All datasets passed validation")
            else:
                logging.warning("⚠️  Some datasets failed validation - check report")

            return all_valid

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test data validation
    train_path = "artifacts/data/train.csv"
    val_path = "artifacts/data/val.csv"
    test_path = "artifacts/data/test.csv"

    validator = DataValidation()
    is_valid = validator.initiate_data_validation(train_path, val_path, test_path)

    print(f"\n{'='*70}")
    print(f"VALIDATION RESULT: {'✅ PASSED' if is_valid else '❌ FAILED'}")
    print(f"{'='*70}\n")

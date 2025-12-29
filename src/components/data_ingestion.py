import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field

from src.exception import CustomException
from src.logger import logging


from src.utils import read_yaml_config

@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion paths."""
    
    config: dict = field(default_factory=lambda: read_yaml_config("config.yaml"))
    
    def __post_init__(self):
        artifacts_config = self.config['artifacts']['data_ingestion']
        self.raw_data_path = artifacts_config['raw_data_path']
        self.train_data_path = artifacts_config['train_data_path']
        self.val_data_path = artifacts_config['val_data_path']
        self.test_data_path = artifacts_config['test_data_path']
        self.source_data_path = artifacts_config['source_data_path']


class DataIngestion:
    """
    Handles data loading and splitting for fraud detection.
    Splits data into 70% train, 15% validation, 15% test.
    """

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, data_path=None):
        """
        Load raw data and split into train/val/test sets.
        
        Args:
            data_path (str, optional): Overrides config source path for testing.

        Returns:
            tuple: Paths to train, val, and test CSV files
        """
        logging.info("Entered data ingestion method")
        if data_path is None:
            data_path = self.ingestion_config.source_data_path

        try:
            # Read dataset
            logging.info(f"Reading dataset from: {data_path}")
            df = pd.read_csv(data_path)
            logging.info(f"Dataset loaded successfully. Shape: {df.shape}")

            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw data backup
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved to: {self.ingestion_config.raw_data_path}")

            # Check fraud distribution
            fraud_rate = df["isFraud"].mean()
            logging.info(f"Fraud rate in dataset: {fraud_rate*100:.4f}%")

            # First split: 70% train, 30% temp (for val + test)
            logging.info("Initiating train-test split (70-30)")
            train_set, temp_set = train_test_split(
                df, test_size=0.30, random_state=42, stratify=df["isFraud"]  # Maintain fraud ratio
            )

            # Second split: Split temp into 50-50 (15% val, 15% test of original)
            logging.info("Splitting remaining 30% into validation and test (15-15)")
            val_set, test_set = train_test_split(
                temp_set, test_size=0.50, random_state=42, stratify=temp_set["isFraud"]
            )

            # Save splits
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            val_set.to_csv(self.ingestion_config.val_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            # Log split statistics
            logging.info(
                f"Train set: {len(train_set)} samples ({len(train_set)/len(df)*100:.1f}%) - Fraud rate: {train_set['isFraud'].mean()*100:.4f}%"
            )
            logging.info(
                f"Val set: {len(val_set)} samples ({len(val_set)/len(df)*100:.1f}%) - Fraud rate: {val_set['isFraud'].mean()*100:.4f}%"
            )
            logging.info(
                f"Test set: {len(test_set)} samples ({len(test_set)/len(df)*100:.1f}%) - Fraud rate: {test_set['isFraud'].mean()*100:.4f}%"
            )

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.val_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test data ingestion
    obj = DataIngestion()
    train_path, val_path, test_path = obj.initiate_data_ingestion()

    print(f"\n{'='*70}")
    print("DATA INGESTION TEST COMPLETED")
    print(f"{'='*70}")
    print(f"Train data: {train_path}")
    print(f"Val data: {val_path}")
    print(f"Test data: {test_path}")
    print(f"{'='*70}\n")

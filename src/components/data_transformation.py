import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation artifacts."""
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for fraud detection feature engineering.
    Implements all engineered features from the EDA notebook.
    """
    
    def fit(self, X, y=None):
        """No fitting required for feature engineering."""
        return self
    
    def transform(self, X):
        """
        Apply feature engineering transformations.
        
        Args:
            X (pd.DataFrame): Input dataframe with raw features
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        try:
            logging.info("Starting feature engineering transformations")
            
            # Create a copy to avoid modifying original
            df = X.copy()
            
            # === 1. TEMPORAL FEATURES ===
            logging.info("Creating temporal features")
            df['hour'] = df['step'] % 24
            df['day'] = (df['step'] // 24).astype(int)
            
            # === 2. BALANCE ERROR FEATURES ===
            logging.info("Creating balance error features")
            df['errorbalanceOrig'] = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']
            df['errorbalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
            
            # Error to amount ratio
            total_error = np.abs(df['errorbalanceOrig']) + np.abs(df['errorbalanceDest'])
            df['errortoamountratio'] = np.where(
                df['amount'] > 0,
                total_error / df['amount'],
                0
            )
            
            # Origin account drained flag
            df['isorigindrained'] = (
                (df['oldbalanceOrg'] > 0) & 
                (df['newbalanceOrig'] == 0)
            ).astype(int)
            
            # === 3. AMOUNT FEATURES ===
            logging.info("Creating amount features")
            df['amountlog'] = np.log1p(df['amount'])
            
            # Amount to balance ratio
            df['amounttobalanceratio'] = np.where(
                df['oldbalanceOrg'] > 0,
                df['amount'] / df['oldbalanceOrg'],
                0
            )
            
            # High amount flag (>= 200k threshold)
            df['ishighamount'] = (df['amount'] >= 200000).astype(int)
            
            # === 4. PATTERN FEATURES ===
            logging.info("Creating pattern features")
            
            # Merchant destination flag
            df['ismerchantdest'] = df['nameDest'].str.startswith('M').astype(int)
            
            # Night transaction flag (hours 0-5)
            df['isnighttransaction'] = df['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
            
            # High-risk transaction type
            df['ishighrisktype'] = df['type'].isin(['TRANSFER', 'CASH_OUT']).astype(int)
            
            # === 5. INTERACTION FEATURES ===
            logging.info("Creating interaction features")
            
            # High-risk type during night
            df['highrisknight'] = (
                df['ishighrisktype'] & df['isnighttransaction']
            ).astype(int)
            
            # High amount to customer (not merchant)
            df['highamountc2c'] = (
                (df['ishighamount'] == 1) & 
                (df['ismerchantdest'] == 0)
            ).astype(int)
            
            # Account drained to non-merchant
            df['drainednonmerchant'] = (
                (df['isorigindrained'] == 1) & 
                (df['ismerchantdest'] == 0)
            ).astype(int)
            
            logging.info(f"Feature engineering completed. Total features: {df.shape[1]}")
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)


class DataTransformation:
    """
    Handles data transformation for fraud detection.
    Applies feature engineering and preprocessing pipeline.
    """
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        """
        Create the preprocessing pipeline.
        
        Returns:
            ColumnTransformer: Sklearn preprocessing pipeline
        """
        try:
            logging.info("Creating data transformation pipeline")
            
            # Features to drop (identifiers and target)
            drop_features = ['nameOrig', 'nameDest', 'isFlaggedFraud']
            
            # Categorical feature (will be one-hot encoded)
            categorical_features = ['type']
            
            # Numerical features (will be scaled)
            # After feature engineering, these are the numeric columns
            numerical_features = [
                'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                'oldbalanceDest', 'newbalanceDest',
                # Engineered features
                'hour', 'day', 'errorbalanceOrig', 'errorbalanceDest',
                'errortoamountratio', 'isorigindrained', 'amountlog',
                'amounttobalanceratio', 'ishighamount', 'ismerchantdest',
                'isnighttransaction', 'ishighrisktype', 'highrisknight',
                'highamountc2c', 'drainednonmerchant'
            ]
            
            # Create pipelines
            numerical_pipeline = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            
            categorical_pipeline = Pipeline(steps=[
                ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ])
            
            # Combine pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_pipeline, numerical_features),
                    ('cat', categorical_pipeline, categorical_features)
                ],
                remainder='drop'  # Drop columns not in numerical or categorical
            )
            
            logging.info("Data transformation pipeline created successfully")
            logging.info(f"Numerical features: {len(numerical_features)}")
            logging.info(f"Categorical features: {len(categorical_features)}")
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, val_path, test_path):
        """
        Apply feature engineering and preprocessing to train/val/test sets.
        
        Args:
            train_path (str): Path to training CSV
            val_path (str): Path to validation CSV
            test_path (str): Path to test CSV
            
        Returns:
            tuple: (train_array, val_array, test_array, preprocessor_path)
        """
        try:
            logging.info("Starting data transformation")
            
            # Read datasets
            logging.info(f"Reading training data from: {train_path}")
            train_df = pd.read_csv(train_path)
            
            logging.info(f"Reading validation data from: {val_path}")
            val_df = pd.read_csv(val_path)
            
            logging.info(f"Reading test data from: {test_path}")
            test_df = pd.read_csv(test_path)
            
            logging.info(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")
            
            # === STEP 1: Feature Engineering ===
            logging.info("Applying feature engineering to all datasets")
            
            feature_engineer = FeatureEngineer()
            train_df = feature_engineer.transform(train_df)
            val_df = feature_engineer.transform(val_df)
            test_df = feature_engineer.transform(test_df)
            
            logging.info(f"After feature engineering - Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
            
            # === STEP 2: Separate features and target ===
            target_column = 'isFraud'
            
            # Training data
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            
            # Validation data
            X_val = val_df.drop(columns=[target_column])
            y_val = val_df[target_column]
            
            # Test data
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            
            logging.info(f"Separated features and target. X_train shape: {X_train.shape}")
            
            # === STEP 3: Get preprocessing pipeline ===
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            
            # === STEP 4: Fit and transform ===
            logging.info("Fitting preprocessor on training data")
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            
            logging.info("Transforming validation data")
            X_val_arr = preprocessing_obj.transform(X_val)
            
            logging.info("Transforming test data")
            X_test_arr = preprocessing_obj.transform(X_test)
            
            logging.info(f"Transformed shapes - Train: {X_train_arr.shape}, Val: {X_val_arr.shape}, Test: {X_test_arr.shape}")
            
            # === STEP 5: Combine features with target ===
            train_arr = np.c_[X_train_arr, np.array(y_train)]
            val_arr = np.c_[X_val_arr, np.array(y_val)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]
            
            # === STEP 6: Save preprocessor ===
            logging.info(f"Saving preprocessing object to: {self.data_transformation_config.preprocessor_obj_file_path}")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("Data transformation completed successfully")
            
            return (
                train_arr,
                val_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test data transformation
    from src.components.data_ingestion import DataIngestion
    
    print("\n" + "="*70)
    print("TESTING DATA TRANSFORMATION")
    print("="*70 + "\n")
    
    # Step 1: Ingest data
    print("Step 1: Data Ingestion...")
    ingestion = DataIngestion()
    train_path, val_path, test_path = ingestion.initiate_data_ingestion()
    print("✅ Data ingestion completed\n")
    
    # Step 2: Transform data
    print("Step 2: Data Transformation...")
    transformation = DataTransformation()
    train_arr, val_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
        train_path, val_path, test_path
    )
    
    print("\n" + "="*70)
    print("TRANSFORMATION TEST COMPLETED")
    print("="*70)
    print(f"Train array shape: {train_arr.shape}")
    print(f"Val array shape: {val_arr.shape}")
    print(f"Test array shape: {test_arr.shape}")
    print(f"Preprocessor saved at: {preprocessor_path}")
    print("="*70 + "\n")

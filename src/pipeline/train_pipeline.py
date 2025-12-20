import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


class TrainingPipeline:
    """
    Complete end-to-end training pipeline for fraud detection.
    Orchestrates: ingestion → validation → transformation → training → evaluation
    """
    
    def __init__(self):
        pass
    
    def start_training(self, data_path='notebook/data/paysim_fraud_data.csv'):
        """
        Execute complete training pipeline.
        
        Args:
            data_path: Path to raw dataset
            
        Returns:
            dict: Training results and metrics
        """
        try:
            logging.info("\n" + "="*70)
            logging.info("STARTING FRAUD DETECTION TRAINING PIPELINE")
            logging.info("="*70 + "\n")
            
            # ====================
            # STEP 1: DATA INGESTION
            # ====================
            logging.info(">>> STEP 1: DATA INGESTION")
            data_ingestion = DataIngestion()
            train_path, val_path, test_path = data_ingestion.initiate_data_ingestion(data_path)
            logging.info("✅ Data ingestion completed\n")
            
            # ====================
            # STEP 2: DATA VALIDATION
            # ====================
            logging.info(">>> STEP 2: DATA VALIDATION")
            data_validation = DataValidation()
            is_valid = data_validation.initiate_data_validation(train_path, val_path, test_path)
            
            if not is_valid:
                logging.warning("⚠️  Data validation warnings found - proceeding with caution\n")
            else:
                logging.info("✅ Data validation completed\n")
            
            # ====================
            # STEP 3: DATA TRANSFORMATION
            # ====================
            logging.info(">>> STEP 3: DATA TRANSFORMATION")
            data_transformation = DataTransformation()
            train_arr, val_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_path, val_path, test_path
            )
            logging.info("✅ Data transformation completed\n")
            
            # ====================
            # STEP 4: MODEL TRAINING
            # ====================
            logging.info(">>> STEP 4: MODEL TRAINING")
            model_trainer = ModelTrainer()
            best_model_name, best_model_score, best_model = model_trainer.initiate_model_trainer(
                train_arr, val_arr
            )
            logging.info("✅ Model training completed\n")
            
            # ====================
            # STEP 5: MODEL EVALUATION
            # ====================
            logging.info(">>> STEP 5: MODEL EVALUATION")
            model_evaluator = ModelEvaluation()
            evaluation_results = model_evaluator.initiate_model_evaluation(
                model_path='artifacts/model.pkl',
                preprocessor_path=preprocessor_path,
                test_path=test_path
            )
            logging.info("✅ Model evaluation completed\n")
            
            # ====================
            # PIPELINE SUMMARY
            # ====================
            logging.info("\n" + "="*70)
            logging.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logging.info("="*70)
            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Validation PR-AUC: {best_model_score:.4f}")
            logging.info(f"Test PR-AUC: {evaluation_results['metrics']['pr_auc']:.4f}")
            logging.info(f"Test F1-Score: {evaluation_results['metrics']['f1_score']:.4f}")
            logging.info(f"Net Business Savings: ${evaluation_results['business_impact']['net_savings']:,.0f}")
            logging.info(f"ROI: {evaluation_results['business_impact']['roi']:.2f}%")
            logging.info("="*70 + "\n")
            
            return {
                'best_model_name': best_model_name,
                'validation_score': best_model_score,
                'test_results': evaluation_results
            }
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Run complete training pipeline
    pipeline = TrainingPipeline()
    results = pipeline.start_training()
    
    print("\n🎉 Training pipeline executed successfully!")
    print(f"Check logs/ folder for detailed execution logs")
    print(f"Check artifacts/ folder for saved models and reports")

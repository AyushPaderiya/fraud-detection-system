from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def test_model_trainer_returns_best():
    ingestion = DataIngestion()
    train_path, val_path, test_path = ingestion.initiate_data_ingestion()
    
    transformation = DataTransformation()
    train_arr, val_arr, test_arr, _ = transformation.initiate_data_transformation(
        train_path, val_path, test_path
    )
    
    trainer = ModelTrainer()
    best_name, best_score, best_model = trainer.initiate_model_trainer(train_arr, val_arr)
    
    assert isinstance(best_name, str)
    assert best_score > 0
    assert best_model is not None

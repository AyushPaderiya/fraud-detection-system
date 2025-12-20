from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

def test_data_transformation_shapes():
    ingestion = DataIngestion()
    train_path, val_path, test_path = ingestion.initiate_data_ingestion()
    
    transformation = DataTransformation()
    train_arr, val_arr, test_arr, _ = transformation.initiate_data_transformation(
        train_path, val_path, test_path
    )
    
    # last column is target
    assert train_arr.shape[1] == val_arr.shape[1] == test_arr.shape[1]

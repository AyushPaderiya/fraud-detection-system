# test_pipeline_components.py
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation

def test_data_pipeline():
    """Test data ingestion and validation."""
    
    print("\n" + "="*70)
    print("TESTING DATA PIPELINE COMPONENTS")
    print("="*70 + "\n")
    
    # Step 1: Data Ingestion
    print("Step 1: Testing Data Ingestion...")
    ingestion = DataIngestion()
    train_path, val_path, test_path = ingestion.initiate_data_ingestion()
    print("✅ Data Ingestion completed\n")
    
    # Step 2: Data Validation
    print("Step 2: Testing Data Validation...")
    validation = DataValidation()
    is_valid = validation.initiate_data_validation(train_path, val_path, test_path)
    
    if is_valid:
        print("\n✅ All tests passed! Pipeline components working correctly.")
    else:
        print("\n⚠️  Validation warnings found - check the report.")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    test_data_pipeline()

from src.pipeline.train_pipeline import TrainingPipeline


def test_full_training_pipeline():
    pipeline = TrainingPipeline()
    results = pipeline.start_training()

    assert "best_model_name" in results
    assert "validation_score" in results
    assert "test_results" in results
    assert results["test_results"]["metrics"]["pr_auc"] > 0

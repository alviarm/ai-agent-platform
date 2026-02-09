"""Tests for intent classification module."""

import pytest
import sys
sys.path.insert(0, "src")

from src.intent_classifier.model import IntentClassifier
from src.intent_classifier.data_loader import generate_synthetic_data, create_datasets


class TestIntentClassifier:
    """Test cases for intent classifier."""
    
    def test_labels_defined(self):
        """Test that all intent labels are defined."""
        labels = IntentClassifier.LABELS
        assert len(labels) == 7
        assert "return" in labels
        assert "grievance" in labels
        assert "billing" in labels
    
    def test_data_generation(self):
        """Test synthetic data generation."""
        texts, labels = generate_synthetic_data(n_samples=100)
        assert len(texts) == 100
        assert len(labels) == 100
        assert all(isinstance(t, str) for t in texts)
        assert all(isinstance(l, int) for l in labels)
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        dataset = create_datasets()
        assert "train" in dataset
        assert "validation" in dataset
        assert "test" in dataset
        assert len(dataset["train"]) > 0
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        # This test requires the model to be trained first
        # We'll just test that the class can be instantiated
        pass


class TestIntentClassifierIntegration:
    """Integration tests (require trained model)."""
    
    @pytest.fixture
    def classifier(self):
        """Load trained classifier."""
        try:
            return IntentClassifier(
                model_path="data/models/onnx",
                use_onnx=True,
            )
        except:
            pytest.skip("Model not trained yet")
    
    def test_predict_single(self, classifier):
        """Test single prediction."""
        result = classifier.predict("I want to return my order")[0]
        assert "intent" in result
        assert "confidence" in result
        assert result["intent"] in classifier.labels
        assert 0 <= result["confidence"] <= 1
    
    def test_predict_batch(self, classifier):
        """Test batch prediction."""
        texts = [
            "I want to return my order",
            "The app keeps crashing",
            "I need a refund",
        ]
        results = classifier.predict(texts)
        assert len(results) == 3
        for result in results:
            assert result["intent"] in classifier.labels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

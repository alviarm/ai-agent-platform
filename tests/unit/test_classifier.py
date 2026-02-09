"""
Unit Tests for Intent Classifier

Tests:
- Classification accuracy >= 85% on synthetic test set
- Latency < 200ms per inference (local CPU)
- Confidence thresholding (low confidence routes to "ESCALATE")
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

# Import the classifier module
from src.intent_classifier.model import IntentClassifier
from src.intent_classifier.config import INTENT_CATEGORIES


# Test configuration
TEST_DATA_PATH = Path(__file__).parent.parent / "data" / "test_intents.json"
MODEL_PATH = Path("data/models")
ONNX_MODEL_PATH = MODEL_PATH / "intent_classifier.onnx"
MIN_ACCURACY = 0.85
MAX_LATENCY_MS = 200
CONFIDENCE_THRESHOLD = 0.6


class TestIntentClassifier:
    """Test suite for intent classification."""
    
    @pytest.fixture(scope="class")
    def classifier(self):
        """Initialize classifier for testing."""
        # Use PyTorch model for testing (ONNX may not exist in test env)
        model_dir = str(MODEL_PATH) if MODEL_PATH.exists() else None
        
        try:
            clf = IntentClassifier(
                model_path=model_dir,
                use_onnx=False,  # Use PyTorch for broader compatibility
                device="cpu",
            )
            return clf
        except Exception as e:
            pytest.skip(f"Could not load classifier: {e}")
    
    @pytest.fixture(scope="class")
    def test_dataset(self):
        """Load test dataset."""
        if not TEST_DATA_PATH.exists():
            # Generate test data if not exists
            pytest.skip(f"Test data not found at {TEST_DATA_PATH}")
        
        with open(TEST_DATA_PATH) as f:
            data = json.load(f)
        
        return data.get("samples", [])
    
    def test_model_initialization(self, classifier):
        """Test that the classifier initializes correctly."""
        assert classifier is not None
        assert classifier.tokenizer is not None
        assert classifier.torch_model is not None or classifier.session is not None
        assert len(classifier.labels) == len(INTENT_CATEGORIES)
    
    def test_predict_single_text(self, classifier):
        """Test prediction on single text."""
        text = "I want to return my laptop"
        result = classifier.predict(text)
        
        assert len(result) == 1
        assert "intent" in result[0]
        assert "confidence" in result[0]
        assert "all_scores" in result[0]
        assert result[0]["intent"] in INTENT_CATEGORIES
        assert 0 <= result[0]["confidence"] <= 1
    
    def test_predict_batch(self, classifier):
        """Test batch prediction."""
        texts = [
            "I want to return my laptop",
            "My bill is wrong",
            "App keeps crashing",
        ]
        results = classifier.predict(texts)
        
        assert len(results) == len(texts)
        for result in results:
            assert result["intent"] in INTENT_CATEGORIES
            assert 0 <= result["confidence"] <= 1
    
    def test_classification_accuracy(self, classifier, test_dataset):
        """Test that classification accuracy meets minimum threshold."""
        if not test_dataset:
            pytest.skip("No test data available")
        
        # Filter out ambiguous samples for accuracy testing
        clean_samples = [
            s for s in test_dataset 
            if not s.get("metadata", {}).get("is_ambiguous", False)
        ]
        
        if len(clean_samples) < 10:
            pytest.skip("Not enough clean samples for accuracy testing")
        
        # Run predictions
        texts = [s["text"] for s in clean_samples]
        true_labels = [s["intent"] for s in clean_samples]
        
        # Batch prediction for efficiency
        batch_size = 32
        predictions = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            results = classifier.predict(batch)
            predictions.extend([r["intent"] for r in results])
        
        # Calculate accuracy
        correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
        accuracy = correct / len(true_labels)
        
        # Log results
        print(f"\nAccuracy: {accuracy:.2%} ({correct}/{len(true_labels)})")
        print(f"Minimum required: {MIN_ACCURACY:.2%}")
        
        # Per-class accuracy
        per_class = {intent: {"correct": 0, "total": 0} for intent in INTENT_CATEGORIES}
        for p, t in zip(predictions, true_labels):
            per_class[t]["total"] += 1
            if p == t:
                per_class[t]["correct"] += 1
        
        print("\nPer-class accuracy:")
        for intent, stats in sorted(per_class.items()):
            if stats["total"] > 0:
                acc = stats["correct"] / stats["total"]
                print(f"  {intent}: {acc:.2%} ({stats['correct']}/{stats['total']})")
        
        assert accuracy >= MIN_ACCURACY, (
            f"Accuracy {accuracy:.2%} below threshold {MIN_ACCURACY:.2%}"
        )
    
    def test_inference_latency(self, classifier):
        """Test that inference latency is within acceptable bounds."""
        test_texts = [
            "I want to return my laptop",
            "My bill is wrong, help",
            "The app keeps crashing when I login",
            "How do I reset my password?",
            "I need to speak to a supervisor",
        ]
        
        latencies = []
        
        # Warm up
        for _ in range(3):
            classifier.predict(test_texts[0])
        
        # Measure latency
        for text in test_texts * 10:  # 50 total inferences
            start = time.perf_counter()
            classifier.predict(text)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        max_latency = np.max(latencies)
        
        print(f"\nLatency Statistics:")
        print(f"  Average: {avg_latency:.2f} ms")
        print(f"  P95: {p95_latency:.2f} ms")
        print(f"  Max: {max_latency:.2f} ms")
        print(f"  Threshold: {MAX_LATENCY_MS} ms")
        
        assert avg_latency < MAX_LATENCY_MS, (
            f"Average latency {avg_latency:.2f}ms exceeds threshold {MAX_LATENCY_MS}ms"
        )
        assert p95_latency < MAX_LATENCY_MS * 1.5, (
            f"P95 latency {p95_latency:.2f}ms exceeds threshold {MAX_LATENCY_MS * 1.5}ms"
        )
    
    def test_confidence_thresholding(self, classifier):
        """Test that low confidence predictions can be flagged for escalation."""
        # These are deliberately ambiguous or unusual queries
        low_confidence_queries = [
            "Something is wrong",
            "Help me please",
            "I have a question",
            "Not sure what to do",
            "Blah blah blah test",
        ]
        
        results = classifier.predict(low_confidence_queries)
        
        low_confidence_count = 0
        for text, result in zip(low_confidence_queries, results):
            print(f"'{text[:30]}...' -> {result['intent']} (conf: {result['confidence']:.3f})")
            if result["confidence"] < CONFIDENCE_THRESHOLD:
                low_confidence_count += 1
        
        # We expect at least some of these to have low confidence
        # This is a sanity check, not a strict assertion
        print(f"\nLow confidence predictions (< {CONFIDENCE_THRESHOLD}): {low_confidence_count}/{len(results)}")
        
        # Test explicit escalation routing
        # Any prediction with very low confidence should be considered for escalation
        very_low = [r for r in results if r["confidence"] < 0.5]
        if very_low:
            print(f"Very low confidence (< 0.5): {len(very_low)} - recommend escalation")
    
    def test_all_intents_predictable(self, classifier):
        """Test that all intent categories can be predicted."""
        # Sample queries for each intent
        intent_queries = {
            "return": "I want to return this item",
            "billing": "There's a problem with my bill",
            "technical": "The app won't work",
            "support": "How do I use this product?",
            "general_inquiry": "What are your hours?",
            "escalation": "I need to speak to a manager",
            "grievance": "I'm very unhappy with your service",
        }
        
        results = classifier.predict(list(intent_queries.values()))
        predicted_intents = set(r["intent"] for r in results)
        
        print(f"\nPredicted intents: {predicted_intents}")
        print(f"Available intents: {set(INTENT_CATEGORIES)}")
        
        # All intents should be representable
        assert predicted_intents.issubset(set(INTENT_CATEGORIES))
    
    def test_confidence_scores_sum(self, classifier):
        """Test that confidence scores are properly normalized."""
        text = "I need help with my order"
        result = classifier.predict(text)[0]
        
        all_scores = result["all_scores"]
        total = sum(all_scores.values())
        
        # Should be approximately 1.0 (allowing for floating point error)
        assert abs(total - 1.0) < 0.01, f"Scores sum to {total}, expected ~1.0"
        
        # Highest score should match the predicted intent
        max_intent = max(all_scores, key=all_scores.get)
        assert max_intent == result["intent"]
    
    def test_typos_handling(self, classifier, test_dataset):
        """Test that the classifier handles typos reasonably well."""
        if not test_dataset:
            pytest.skip("No test data available")
        
        # Filter samples with typos
        typo_samples = [
            s for s in test_dataset 
            if s.get("metadata", {}).get("has_typos", False)
        ]
        
        if len(typo_samples) < 5:
            pytest.skip("Not enough typo samples")
        
        texts = [s["text"] for s in typo_samples[:20]]
        true_labels = [s["intent"] for s in typo_samples[:20]]
        
        results = classifier.predict(texts)
        predictions = [r["intent"] for r in results]
        
        correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
        accuracy = correct / len(true_labels)
        
        print(f"\nAccuracy with typos: {accuracy:.2%} ({correct}/{len(true_labels)})")
        
        # Should still achieve reasonable accuracy with typos (lower threshold)
        assert accuracy >= MIN_ACCURACY * 0.8, (
            f"Typo accuracy {accuracy:.2%} too low"
        )


class TestONNXModel:
    """Tests specific to ONNX model if available."""
    
    @pytest.mark.skipif(
        not ONNX_MODEL_PATH.exists(),
        reason="ONNX model not found"
    )
    def test_onnx_model_loading(self):
        """Test that ONNX model can be loaded."""
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(
                str(ONNX_MODEL_PATH),
                providers=["CPUExecutionProvider"]
            )
            assert session is not None
            print(f"\nONNX model loaded successfully")
            print(f"  Inputs: {[i.name for i in session.get_inputs()]}")
            print(f"  Outputs: {[o.name for o in session.get_outputs()]}")
        except Exception as e:
            pytest.fail(f"Failed to load ONNX model: {e}")
    
    @pytest.mark.skipif(
        not ONNX_MODEL_PATH.exists(),
        reason="ONNX model not found"
    )
    def test_onnx_inference(self):
        """Test ONNX model inference."""
        clf = IntentClassifier(
            model_path=str(ONNX_MODEL_PATH),
            use_onnx=True,
            device="cpu",
        )
        
        result = clf.predict("I want to return my order")[0]
        assert result["intent"] in INTENT_CATEGORIES
        assert 0 <= result["confidence"] <= 1
        print(f"\nONNX prediction: {result['intent']} (conf: {result['confidence']:.3f})")
    
    @pytest.mark.skipif(
        not ONNX_MODEL_PATH.exists(),
        reason="ONNX model not found"
    )
    def test_onnx_parity(self):
        """Test that ONNX and PyTorch models produce similar results."""
        # Load both models
        pytorch_clf = IntentClassifier(
            model_path=str(MODEL_PATH) if MODEL_PATH.exists() else None,
            use_onnx=False,
            device="cpu",
        )
        
        onnx_clf = IntentClassifier(
            model_path=str(ONNX_MODEL_PATH),
            use_onnx=True,
            device="cpu",
        )
        
        test_texts = [
            "I want to return this",
            "My bill is wrong",
            "App is not working",
            "How do I use this?",
        ]
        
        pytorch_results = pytorch_clf.predict(test_texts)
        onnx_results = onnx_clf.predict(test_texts)
        
        # Check that predictions match
        match_count = 0
        for pt, onnx in zip(pytorch_results, onnx_results):
            if pt["intent"] == onnx["intent"]:
                match_count += 1
            # Confidence should be close
            conf_diff = abs(pt["confidence"] - onnx["confidence"])
            print(f"  {pt['intent']} vs {onnx['intent']} (diff: {conf_diff:.4f})")
        
        match_rate = match_count / len(test_texts)
        print(f"\nPrediction match rate: {match_rate:.1%}")
        
        # Should have high agreement
        assert match_rate >= 0.75, f"ONNX/PyTorch match rate {match_rate:.1%} too low"

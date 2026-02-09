"""Tests for feedback analysis pipeline."""

import pytest
import sys
sys.path.insert(0, "src")

from src.feedback_pipeline.preprocessor import TextPreprocessor
from src.feedback_pipeline.analyzer import FeedbackAnalyzer
from src.feedback_pipeline.reporter import FeedbackReporter


class TestTextPreprocessor:
    """Test text preprocessing."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor."""
        return TextPreprocessor(use_lemmatization=False)
    
    def test_preprocess(self, preprocessor):
        """Test basic preprocessing."""
        text = "Check out https://example.com or email me@test.com! Order #ABC12345"
        result = preprocessor.preprocess(text)
        assert "https://" not in result
        assert "@" not in result
        assert "ABC12345" not in result
    
    def test_tokenize(self, preprocessor):
        """Test tokenization."""
        tokens = preprocessor.tokenize("Hello world!")
        assert "Hello" in tokens
        assert "world" in tokens
    
    def test_process(self, preprocessor):
        """Test full processing."""
        result = preprocessor.process("Hello WORLD! Visit https://test.com")
        assert "hello" in result
        assert "world" in result
        assert "https" not in result


class TestFeedbackAnalyzer:
    """Test feedback analysis."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        return FeedbackAnalyzer(use_transformer_sentiment=False)
    
    def test_sentiment_positive(self, analyzer):
        """Test positive sentiment detection."""
        result = analyzer.analyze_sentiment("This was excellent service!")
        assert result["label"] == "positive"
    
    def test_sentiment_negative(self, analyzer):
        """Test negative sentiment detection."""
        result = analyzer.analyze_sentiment("This is terrible and frustrating")
        assert result["label"] == "negative"
    
    def test_extract_keywords(self, analyzer):
        """Test keyword extraction."""
        text = "The customer service was very helpful and the product quality is excellent"
        keywords = analyzer.extract_keywords_rake(text, num_keywords=5)
        assert len(keywords) <= 5
        # Keywords should be phrases
        assert all(isinstance(k, tuple) for k in keywords)
    
    def test_analyze_feedback(self, analyzer):
        """Test full feedback analysis."""
        result = analyzer.analyze_feedback(
            "The support team was very helpful!",
            metadata={"conversation_id": "test-123"},
        )
        assert "sentiment" in result
        assert "keywords" in result
        assert "metadata" in result


class TestFeedbackReporter:
    """Test feedback reporting."""
    
    @pytest.fixture
    def reporter(self):
        """Create reporter with sample data."""
        results = [
            {
                "original_text": "Great service!",
                "processed_text": "great service",
                "sentiment": {"label": "positive", "confidence": 0.9},
                "keywords": [{"phrase": "great service", "score": 1.0}],
                "metadata": {"intent": "support"},
            },
            {
                "original_text": "Terrible experience",
                "processed_text": "terrible experience",
                "sentiment": {"label": "negative", "confidence": 0.8},
                "keywords": [{"phrase": "terrible experience", "score": 1.0}],
                "metadata": {"intent": "grievance"},
            },
        ]
        return FeedbackReporter(results)
    
    def test_generate_summary(self, reporter):
        """Test summary generation."""
        summary = reporter.generate_summary()
        assert summary["total_feedback"] == 2
        assert "sentiment_distribution" in summary
        assert "top_keywords" in summary
    
    def test_generate_intent_report(self, reporter):
        """Test intent-based report."""
        report = reporter.generate_intent_report()
        assert "support" in report
        assert "grievance" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

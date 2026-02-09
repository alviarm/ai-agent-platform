"""Analyzer for Lambda (simplified)."""

from typing import Dict, List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class FeedbackAnalyzer:
    """Simple feedback analyzer."""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment."""
        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]
        
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "label": sentiment,
            "confidence": abs(compound),
            "scores": scores,
        }

"""Feedback analysis with sentiment, keywords, and topics."""

import json
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .config import (
    CUSTOM_STOP_WORDS,
    LDA_ITERATIONS,
    LDA_PASSES,
    NUM_TOPICS,
    RAKE_MAX_WORDS,
    RAKE_MIN_FREQUENCY,
    SENTIMENT_MODEL,
    TFIDF_MAX_FEATURES,
    USE_VADER_FALLBACK,
)
from .preprocessor import TextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackAnalyzer:
    """Analyzer for customer feedback with sentiment, keywords, and topics."""
    
    def __init__(
        self,
        use_transformer_sentiment: bool = True,
        num_topics: int = NUM_TOPICS,
    ):
        """Initialize analyzer.
        
        Args:
            use_transformer_sentiment: Whether to use transformer-based sentiment
            num_topics: Number of topics for LDA
        """
        self.preprocessor = TextPreprocessor()
        self.num_topics = num_topics
        
        # Sentiment analyzers
        self.use_transformer = use_transformer_sentiment
        if use_transformer_sentiment:
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=SENTIMENT_MODEL,
                    tokenizer=SENTIMENT_MODEL,
                    device=-1,  # CPU
                )
            except Exception as e:
                logger.warning(f"Failed to load transformer sentiment model: {e}")
                self.use_transformer = False
        
        if not self.use_transformer or USE_VADER_FALLBACK:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Keyword extraction
        self.rake_extractor = Rake(
            max_length=RAKE_MAX_WORDS,
            min_length=1,
        )
        
        # Topic model (will be fitted on data)
        self.lda_model = None
        self.dictionary = None
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text.
        
        Args:
            text: Input text
            
        Returns:
            Sentiment dictionary with label and scores
        """
        # Transformer-based sentiment
        if self.use_transformer:
            try:
                result = self.sentiment_pipeline(text[:512])[0]  # Truncate for model
                label = result["label"].lower()
                score = result["score"]
                
                return {
                    "label": "positive" if label == "positive" else "negative",
                    "confidence": score,
                    "scores": {
                        "positive": score if label == "positive" else 1 - score,
                        "negative": score if label == "negative" else 1 - score,
                    },
                    "method": "transformer",
                }
            except Exception as e:
                logger.warning(f"Transformer sentiment failed: {e}")
        
        # VADER fallback
        scores = self.vader_analyzer.polarity_scores(text)
        compound = scores["compound"]
        
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "label": label,
            "confidence": abs(compound),
            "scores": {
                "positive": scores["pos"],
                "negative": scores["neg"],
                "neutral": scores["neu"],
                "compound": compound,
            },
            "method": "vader",
        }
    
    def extract_keywords_rake(self, text: str, num_keywords: int = 10) -> List[Tuple[str, float]]:
        """Extract keywords using RAKE.
        
        Args:
            text: Input text
            num_keywords: Number of keywords to return
            
        Returns:
            List of (keyword, score) tuples
        """
        self.rake_extractor.extract_keywords_from_text(text)
        keywords = self.rake_extractor.get_ranked_phrases_with_scores()
        return keywords[:num_keywords]
    
    def extract_keywords_tfidf(
        self,
        texts: List[str],
        num_keywords: int = 20,
    ) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF across corpus.
        
        Args:
            texts: List of texts
            num_keywords: Number of keywords to return
            
        Returns:
            List of (keyword, score) tuples
        """
        processed_texts = self.preprocessor.process_batch(texts)
        
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            stop_words="english",
            ngram_range=(1, 2),
        )
        
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get average TF-IDF scores
        scores = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # Sort by score
        keywords = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
        keywords.sort(key=lambda x: x[1], reverse=True)
        
        return keywords[:num_keywords]
    
    def fit_topic_model(self, texts: List[str]) -> LdaModel:
        """Fit LDA topic model on texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Fitted LDA model
        """
        # Preprocess and tokenize
        tokenized_texts = self.preprocessor.process_batch(texts, return_tokens=True)
        
        # Remove empty documents
        tokenized_texts = [t for t in tokenized_texts if t]
        
        # Create dictionary
        self.dictionary = corpora.Dictionary(tokenized_texts)
        
        # Filter extremes
        self.dictionary.filter_extremes(no_below=2, no_above=0.9)
        
        # Create corpus
        corpus = [self.dictionary.doc2bow(text) for text in tokenized_texts]
        
        # Train LDA model
        self.lda_model = LdaModel(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=42,
            passes=LDA_PASSES,
            iterations=LDA_ITERATIONS,
        )
        
        return self.lda_model
    
    def get_topics(self, num_words: int = 10) -> List[Dict]:
        """Get topic descriptions.
        
        Args:
            num_words: Number of words per topic
            
        Returns:
            List of topic dictionaries
        """
        if not self.lda_model:
            return []
        
        topics = []
        for topic_id in range(self.num_topics):
            words = self.lda_model.show_topic(topic_id, num_words)
            topics.append({
                "id": topic_id,
                "words": [{"word": w, "weight": float(p)} for w, p in words],
                "description": ", ".join([w for w, _ in words[:5]]),
            })
        
        return topics
    
    def analyze_document_topics(self, text: str) -> List[Tuple[int, float]]:
        """Get topic distribution for a document.
        
        Args:
            text: Input text
            
        Returns:
            List of (topic_id, probability) tuples
        """
        if not self.lda_model or not self.dictionary:
            return []
        
        tokens = self.preprocessor.process(text, return_tokens=True)
        bow = self.dictionary.doc2bow(tokens)
        
        topic_dist = self.lda_model.get_document_topics(bow)
        return sorted(topic_dist, key=lambda x: x[1], reverse=True)
    
    def analyze_feedback(
        self,
        text: str,
        metadata: Dict = None,
    ) -> Dict:
        """Full analysis of a single feedback item.
        
        Args:
            text: Feedback text
            metadata: Additional metadata
            
        Returns:
            Analysis results dictionary
        """
        # Preprocess
        processed = self.preprocessor.process(text)
        
        # Sentiment
        sentiment = self.analyze_sentiment(text)
        
        # Keywords
        keywords = self.extract_keywords_rake(text)
        
        # Topics (if model fitted)
        topics = self.analyze_document_topics(text) if self.lda_model else []
        
        return {
            "original_text": text,
            "processed_text": processed,
            "sentiment": sentiment,
            "keywords": [{"phrase": k, "score": float(s)} for k, s in keywords],
            "topics": [{"topic_id": t, "probability": float(p)} for t, p in topics],
            "metadata": metadata or {},
        }
    
    def analyze_batch(
        self,
        feedbacks: List[Dict],
    ) -> List[Dict]:
        """Analyze a batch of feedback.
        
        Args:
            feedbacks: List of feedback dictionaries with 'text' key
            
        Returns:
            List of analysis results
        """
        results = []
        for feedback in feedbacks:
            text = feedback.get("text", "")
            metadata = {k: v for k, v in feedback.items() if k != "text"}
            result = self.analyze_feedback(text, metadata)
            results.append(result)
        
        return results

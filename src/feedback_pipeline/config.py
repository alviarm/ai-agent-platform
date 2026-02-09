"""Configuration for Feedback Analysis Pipeline."""

from typing import List

# Sentiment Analysis
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
USE_VADER_FALLBACK = True

# Keyword Extraction
RAKE_MAX_WORDS = 3
RAKE_MIN_FREQUENCY = 1
TFIDF_MAX_FEATURES = 100

# Topic Modeling
NUM_TOPICS = 5
LDA_PASSES = 15
LDA_ITERATIONS = 400

# Data Processing
BATCH_SIZE = 100
DATE_FORMAT = "%Y-%m-%d"

# Aggregation
TREND_WINDOW_DAYS = 7
TOP_KEYWORDS_COUNT = 20

# Stop words for keyword extraction
CUSTOM_STOP_WORDS: List[str] = [
    "customer", "service", "agent", "representative",
    "company", "product", "order", "call", "phone",
    "email", "chat", "support", "help", "issue",
    "problem", "question", "answer", "response",
]

# Intent to category mapping for reporting
INTENT_CATEGORIES = {
    "return": "Returns & Refunds",
    "grievance": "Complaints",
    "billing": "Billing & Payments",
    "technical": "Technical Issues",
    "support": "Product Support",
    "general_inquiry": "General Inquiries",
    "escalation": "Escalations",
}

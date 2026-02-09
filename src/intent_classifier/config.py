"""Configuration for Intent Classification."""

from typing import List

# Intent categories for customer service
INTENT_CATEGORIES: List[str] = [
    "return",
    "grievance", 
    "billing",
    "technical",
    "support",
    "general_inquiry",
    "escalation",
]

# Model configuration
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1

# Paths
MODEL_DIR = "data/models"
ONNX_MODEL_PATH = f"{MODEL_DIR}/intent_classifier.onnx"
LABEL_ENCODER_PATH = f"{MODEL_DIR}/label_encoder.pkl"

# Training data configuration
SYNTHETIC_DATA_SIZE = 5000
TEST_SIZE = 0.2
RANDOM_SEED = 42

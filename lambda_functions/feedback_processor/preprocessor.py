"""Text preprocessor for Lambda (simplified)."""

import re
import string
from typing import List


class TextPreprocessor:
    """Simple text preprocessor."""
    
    def preprocess(self, text: str) -> str:
        """Basic preprocessing."""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.split()

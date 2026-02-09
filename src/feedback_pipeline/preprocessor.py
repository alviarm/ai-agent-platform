"""Text preprocessing for feedback analysis."""

import re
import string
from typing import List

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from .config import CUSTOM_STOP_WORDS

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


class TextPreprocessor:
    """Text preprocessing pipeline for feedback analysis."""
    
    def __init__(
        self,
        use_lemmatization: bool = True,
        remove_stopwords: bool = True,
        custom_stopwords: List[str] = None,
    ):
        """Initialize preprocessor.
        
        Args:
            use_lemmatization: Whether to use spaCy lemmatization
            remove_stopwords: Whether to remove stopwords
            custom_stopwords: Additional stopwords to remove
        """
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        
        # Load spaCy model
        if use_lemmatization:
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            except OSError:
                # Fallback if model not installed
                self.nlp = None
                self.use_lemmatization = False
        else:
            self.nlp = None
        
        # Build stopwords set
        self.stop_words = set()
        if remove_stopwords:
            self.stop_words = set(stopwords.words("english"))
            self.stop_words.update(CUSTOM_STOP_WORDS)
            if custom_stopwords:
                self.stop_words.update(custom_stopwords)
    
    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)
        
        # Remove phone numbers
        text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "", text)
        
        # Remove order numbers (alphanumeric sequences)
        text = re.sub(r"\b[A-Z]{2,}\d{6,}\b", "", text, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^\w\s.,!?-]", "", text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text.
        
        Args:
            text: Preprocessed text
            
        Returns:
            List of tokens
        """
        tokens = word_tokenize(text)
        return tokens
    
    def clean_tokens(self, tokens: List[str]) -> List[str]:
        """Clean and filter tokens.
        
        Args:
            tokens: Raw tokens
            
        Returns:
            Cleaned tokens
        """
        cleaned = []
        for token in tokens:
            # Remove punctuation-only tokens
            if all(c in string.punctuation for c in token):
                continue
            
            # Remove numbers
            if token.isdigit():
                continue
            
            # Check length
            if len(token) < 2:
                continue
            
            # Remove stopwords
            if self.remove_stopwords and token.lower() in self.stop_words:
                continue
            
            cleaned.append(token.lower())
        
        return cleaned
    
    def lemmatize(self, text: str) -> str:
        """Lemmatize text using spaCy.
        
        Args:
            text: Input text
            
        Returns:
            Lemmatized text
        """
        if not self.nlp:
            return text
        
        doc = self.nlp(text)
        lemmas = [token.lemma_ for token in doc if not token.is_punct]
        return " ".join(lemmas)
    
    def process(self, text: str, return_tokens: bool = False):
        """Full processing pipeline.
        
        Args:
            text: Raw text
            return_tokens: Whether to return tokens instead of string
            
        Returns:
            Processed text or tokens
        """
        # Preprocess
        text = self.preprocess(text)
        
        # Lemmatize if enabled
        if self.use_lemmatization:
            text = self.lemmatize(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Clean tokens
        tokens = self.clean_tokens(tokens)
        
        if return_tokens:
            return tokens
        return " ".join(tokens)
    
    def process_batch(
        self,
        texts: List[str],
        return_tokens: bool = False,
    ) -> List:
        """Process a batch of texts.
        
        Args:
            texts: List of raw texts
            return_tokens: Whether to return tokens
            
        Returns:
            List of processed texts or tokens
        """
        return [self.process(text, return_tokens) for text in texts]

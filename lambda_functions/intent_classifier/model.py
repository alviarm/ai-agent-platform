"""Lightweight intent classifier model for Lambda."""

from typing import Dict, List, Union
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class IntentClassifier:
    """ONNX-based intent classifier for Lambda."""
    
    LABELS = [
        "return", "grievance", "billing", "technical",
        "support", "general_inquiry", "escalation",
    ]
    
    def __init__(self, model_path: str = "/tmp/model", use_onnx: bool = True):
        """Initialize classifier."""
        self.labels = self.LABELS
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load ONNX model
        import os
        onnx_path = os.path.join(model_path, "model.onnx")
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
    
    def predict(self, texts: Union[str, List[str]]) -> List[Dict]:
        """Predict intent for input text(s)."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="np",
        )
        
        # Run inference
        outputs = self.session.run(None, {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        })
        
        logits = outputs[0]
        probs = self._softmax(logits)
        predictions = np.argmax(probs, axis=-1)
        
        # Format results
        results = []
        for i, pred in enumerate(predictions):
            intent = self.id2label[pred]
            all_scores = {self.id2label[j]: round(probs[i][j], 4) for j in range(len(self.labels))}
            results.append({
                "intent": intent,
                "confidence": round(probs[i][pred], 4),
                "all_scores": all_scores,
            })
        
        return results
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax values."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

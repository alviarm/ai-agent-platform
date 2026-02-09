"""Intent Classification Model using DistilBERT."""

import pickle
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertForSequenceClassification,
)

from .config import INTENT_CATEGORIES, MAX_LENGTH, MODEL_NAME


class IntentClassifier:
    """BERT-based intent classifier with ONNX inference support."""
    
    def __init__(
        self,
        model_path: str = None,
        use_onnx: bool = True,
        device: str = None,
    ):
        """Initialize the classifier.
        
        Args:
            model_path: Path to model (PyTorch or ONNX)
            use_onnx: Whether to use ONNX runtime for inference
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_onnx = use_onnx and model_path and model_path.endswith(".onnx")
        self.labels = INTENT_CATEGORIES
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        if self.use_onnx:
            self.session = ort.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self.torch_model = None
        else:
            self.session = None
            if model_path and Path(model_path).exists():
                self.torch_model = DistilBertForSequenceClassification.from_pretrained(
                    model_path
                )
            else:
                self.torch_model = DistilBertForSequenceClassification.from_pretrained(
                    MODEL_NAME,
                    num_labels=len(self.labels),
                    id2label=self.id2label,
                    label2id=self.label2id,
                )
            self.torch_model.to(self.device)
            self.torch_model.eval()
    
    def predict(self, texts: Union[str, List[str]]) -> List[Dict[str, Union[str, float]]]:
        """Predict intent for input text(s).
        
        Args:
            texts: Single text or list of texts to classify
            
        Returns:
            List of prediction dictionaries with intent and confidence
        """
        if isinstance(texts, str):
            texts = [texts]
            
        if self.use_onnx:
            return self._predict_onnx(texts)
        else:
            return self._predict_torch(texts)
    
    def _predict_torch(self, texts: List[str]) -> List[Dict]:
        """Predict using PyTorch model."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.torch_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
        
        results = []
        for i, pred in enumerate(predictions):
            intent = self.id2label[pred.item()]
            confidence = probs[i][pred].item()
            all_scores = {
                self.id2label[j]: probs[i][j].item() 
                for j in range(len(self.labels))
            }
            results.append({
                "intent": intent,
                "confidence": confidence,
                "all_scores": all_scores,
            })
        return results
    
    def _predict_onnx(self, texts: List[str]) -> List[Dict]:
        """Predict using ONNX runtime."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np",
        )
        
        onnx_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        
        outputs = self.session.run(None, onnx_inputs)
        logits = outputs[0]
        
        probs = self._softmax(logits)
        predictions = np.argmax(probs, axis=-1)
        
        results = []
        for i, pred in enumerate(predictions):
            intent = self.id2label[pred]
            confidence = probs[i][pred]
            all_scores = {
                self.id2label[j]: probs[i][j] 
                for j in range(len(self.labels))
            }
            results.append({
                "intent": intent,
                "confidence": confidence,
                "all_scores": all_scores,
            })
        return results
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax values."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def save(self, output_dir: str):
        """Save PyTorch model and tokenizer."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if self.torch_model:
            self.torch_model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
    
    @classmethod
    def load(cls, model_dir: str, use_onnx: bool = False) -> "IntentClassifier":
        """Load a saved model."""
        return cls(model_path=model_dir, use_onnx=use_onnx)

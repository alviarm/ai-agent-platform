"""Intent Classification Module for Customer Service AI."""

from .model import IntentClassifier
from .train import train_model
from .evaluate import evaluate_model
from .export_onnx import export_to_onnx

__all__ = ["IntentClassifier", "train_model", "evaluate_model", "export_to_onnx"]

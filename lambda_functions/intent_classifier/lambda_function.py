"""Lambda function for intent classification."""

import json
import logging
import os
from typing import Dict, List

import boto3

# Configure logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

# Import model (will be in same directory in container)
from model import IntentClassifier

# Global instance for Lambda container reuse
_classifier = None
_s3_client = None


def get_classifier() -> IntentClassifier:
    """Get or initialize classifier (with caching)."""
    global _classifier, _s3_client
    
    if _classifier is None:
        logger.info("Initializing classifier...")
        
        # Download model from S3 if needed
        model_path = "/tmp/model"
        os.makedirs(model_path, exist_ok=True)
        
        bucket = os.environ.get("MODEL_BUCKET")
        model_key = os.environ.get("MODEL_KEY", "models/intent_classifier.onnx")
        
        if bucket:
            if _s3_client is None:
                _s3_client = boto3.client("s3")
            
            local_model_path = f"{model_path}/model.onnx"
            if not os.path.exists(local_model_path):
                logger.info(f"Downloading model from s3://{bucket}/{model_key}")
                _s3_client.download_file(bucket, model_key, local_model_path)
        
        _classifier = IntentClassifier(model_path=model_path, use_onnx=True)
        logger.info("Classifier initialized")
    
    return _classifier


def handler(event: Dict, context) -> Dict:
    """Lambda handler for intent classification."""
    logger.info(f"Received event: {json.dumps(event)}")
    
    try:
        # Parse request body
        if "body" in event:
            body = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]
        else:
            body = event
        
        # Get input text
        if "text" in body:
            texts = [body["text"]]
        elif "texts" in body:
            texts = body["texts"]
        else:
            return {
                "statusCode": 400,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                },
                "body": json.dumps({"error": "Missing 'text' or 'texts' field"}),
            }
        
        # Get classifier and predict
        classifier = get_classifier()
        results = classifier.predict(texts)
        
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"results": results, "count": len(results)}),
        }
        
    except Exception as e:
        logger.exception("Error processing request")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"error": str(e), "type": type(e).__name__}),
        }

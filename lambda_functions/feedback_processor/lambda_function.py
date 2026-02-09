"""Lambda function for feedback processing."""

import json
import logging
import os
import uuid
from typing import Dict
from datetime import datetime

import boto3
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rake_nltk import Rake
import nltk

# Download NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir="/tmp/nltk_data")
    nltk.download("stopwords", download_dir="/tmp/nltk_data")

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

_analyzer = None
_dynamodb = None
_feedback_table = None
_analytics_table = None


def get_analyzer():
    """Get sentiment analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def get_tables():
    """Get DynamoDB tables."""
    global _dynamodb, _feedback_table, _analytics_table
    if _dynamodb is None:
        _dynamodb = boto3.resource("dynamodb")
        _feedback_table = _dynamodb.Table(os.environ.get("FEEDBACK_TABLE", "csa-feedback"))
        _analytics_table = _dynamodb.Table(os.environ.get("ANALYTICS_TABLE", "csa-analytics"))
    return _feedback_table, _analytics_table


def analyze_text(text: str) -> Dict:
    """Analyze text for sentiment and keywords."""
    analyzer = get_analyzer()
    
    # Sentiment
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    
    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    # Keywords
    rake = Rake(max_length=3)
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()[:10]
    
    return {
        "sentiment": sentiment,
        "confidence": abs(compound),
        "scores": scores,
        "keywords": keywords,
    }


def handler(event: Dict, context) -> Dict:
    """Lambda handler."""
    logger.info(f"Event: {json.dumps(event)}")
    
    try:
        http_method = event.get("httpMethod", "POST")
        
        if http_method == "POST":
            # Submit feedback
            body = json.loads(event["body"]) if isinstance(event.get("body"), str) else event.get("body", {})
            
            feedback_text = body.get("text", body.get("feedback", ""))
            conversation_id = body.get("conversation_id", "")
            rating = body.get("rating", "neutral")
            
            if not feedback_text:
                return {
                    "statusCode": 400,
                    "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                    "body": json.dumps({"error": "Missing 'text' field"}),
                }
            
            # Analyze
            analysis = analyze_text(feedback_text)
            
            # Store in DynamoDB
            feedback_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()
            
            item = {
                "feedback_id": feedback_id,
                "timestamp": timestamp,
                "conversation_id": conversation_id,
                "text": feedback_text,
                "rating": rating,
                "analysis": analysis,
            }
            
            feedback_table, _ = get_tables()
            feedback_table.put_item(Item=item)
            
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": json.dumps({
                    "feedback_id": feedback_id,
                    "analysis": analysis,
                }),
            }
            
        elif http_method == "GET":
            # Get analytics
            path = event.get("path", "")
            
            if "/analytics" in path:
                _, analytics_table = get_tables()
                response = analytics_table.scan(Limit=100)
                items = response.get("Items", [])
                
                return {
                    "statusCode": 200,
                    "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                    "body": json.dumps({
                        "metrics": items,
                        "count": len(items),
                    }),
                }
            else:
                feedback_table, _ = get_tables()
                query_params = event.get("queryStringParameters") or {}
                conversation_id = query_params.get("conversation_id")
                
                if conversation_id:
                    response = feedback_table.query(
                        IndexName="conversation-index",
                        KeyConditionExpression="conversation_id = :cid",
                        ExpressionAttributeValues={":cid": conversation_id},
                    )
                else:
                    response = feedback_table.scan(Limit=50)
                
                return {
                    "statusCode": 200,
                    "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                    "body": json.dumps({
                        "feedback": response.get("Items", []),
                        "count": len(response.get("Items", [])),
                    }),
                }
        
        return {
            "statusCode": 405,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Method not allowed"}),
        }
        
    except Exception as e:
        logger.exception("Error")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)}),
        }

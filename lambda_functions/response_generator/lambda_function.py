"""Lambda function for response generation."""

import json
import logging
import os
from typing import Dict

import boto3
import openai
from conversation_manager import ConversationManager

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

_conv_manager = None
_secrets_client = None


def get_api_key() -> str:
    """Get OpenAI API key from Secrets Manager."""
    global _secrets_client
    if _secrets_client is None:
        _secrets_client = boto3.client("secretsmanager")
    
    try:
        response = _secrets_client.get_secret_value(
            SecretId=os.environ.get("OPENAI_API_KEY_SECRET", "csa/openai-api-key")
        )
        return json.loads(response["SecretString"])["api_key"]
    except:
        return os.environ.get("OPENAI_API_KEY", "")


def get_conversation_manager() -> ConversationManager:
    """Get or initialize conversation manager."""
    global _conv_manager
    if _conv_manager is None:
        _conv_manager = ConversationManager(
            table_name=os.environ.get("CONVERSATIONS_TABLE", "csa-conversations"),
            use_dynamodb=True,
        )
    return _conv_manager


def get_system_prompt(intent: str) -> str:
    """Get system prompt for intent."""
    prompts = {
        "return": "You are a helpful customer service assistant specializing in returns and refunds.",
        "grievance": "You are a senior customer service representative handling escalated complaints.",
        "billing": "You are a billing specialist helping with payment and refund issues.",
        "technical": "You are a technical support specialist helping customers with product issues.",
        "support": "You are a product support specialist providing helpful information.",
        "general_inquiry": "You are a friendly customer service representative.",
        "escalation": "You are handling a priority case requiring immediate attention.",
    }
    return prompts.get(intent, "You are a helpful customer service AI assistant.")


def call_llm(prompt: str) -> str:
    """Call OpenAI API."""
    openai.api_key = get_api_key()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def handler(event: Dict, context) -> Dict:
    """Lambda handler."""
    logger.info(f"Event: {json.dumps(event)}")
    
    try:
        body = json.loads(event["body"]) if "body" in event and isinstance(event["body"], str) else event.get("body", event)
        
        query = body.get("message", body.get("text", ""))
        conversation_id = body.get("conversation_id")
        intent = body.get("intent", "general_inquiry")
        user_id = body.get("user_id")
        
        if not query:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"error": "Missing 'message' field"}),
            }
        
        conv_manager = get_conversation_manager()
        
        if not conversation_id:
            conversation_id = conv_manager.create_conversation(user_id)
        
        history = conv_manager.get_history(conversation_id)
        
        # Build prompt
        history_str = ""
        for turn in history[-6:]:
            role = "Customer" if turn["role"] == "user" else "Assistant"
            history_str += f"{role}: {turn['content']}\n"
        
        prompt = f"""{get_system_prompt(intent)}

Conversation History:
{history_str}
Customer: {query}

Assistant:"""
        
        try:
            response_text = call_llm(prompt)
            model_used = "gpt-3.5-turbo"
        except Exception as e:
            logger.error(f"LLM failed: {e}")
            response_text = "I apologize, but I'm having trouble. Connecting you with a representative."
            model_used = "fallback"
        
        # Save turns
        conv_manager.add_turn(conversation_id, "user", query, {"intent": intent})
        conv_manager.add_turn(conversation_id, "assistant", response_text, {"model": model_used})
        
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({
                "conversation_id": conversation_id,
                "response": response_text,
                "intent": intent,
                "model_used": model_used,
            }),
        }
        
    except Exception as e:
        logger.exception("Error")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)}),
        }

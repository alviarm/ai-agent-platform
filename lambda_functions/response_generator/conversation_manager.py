"""Conversation state management for Lambda."""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import ClientError


class ConversationManager:
    """Manage conversation state in DynamoDB."""
    
    def __init__(self, table_name: str, use_dynamodb: bool = True):
        self.table_name = table_name
        self.use_dynamodb = use_dynamodb
        self._local_store: Dict[str, Dict] = {}
        
        if use_dynamodb:
            self.dynamodb = boto3.resource("dynamodb")
            self.table = self.dynamodb.Table(table_name)
        else:
            self.dynamodb = None
            self.table = None
    
    def create_conversation(self, user_id: str = None) -> str:
        """Create a new conversation."""
        conversation_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        conversation = {
            "conversation_id": conversation_id,
            "user_id": user_id or "anonymous",
            "created_at": timestamp,
            "updated_at": timestamp,
            "status": "active",
            "intent": None,
            "history": [],
        }
        
        if self.use_dynamodb:
            self.table.put_item(Item=conversation)
        else:
            self._local_store[conversation_id] = conversation
        
        return conversation_id
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get conversation by ID."""
        if self.use_dynamodb:
            try:
                response = self.table.get_item(Key={"conversation_id": conversation_id})
                return response.get("Item")
            except ClientError:
                return None
        return self._local_store.get(conversation_id)
    
    def add_turn(self, conversation_id: str, role: str, content: str, metadata: Dict = None):
        """Add a conversation turn."""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False
        
        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        
        conversation["history"].append(turn)
        conversation["updated_at"] = datetime.utcnow().isoformat()
        
        if metadata and "intent" in metadata:
            conversation["intent"] = metadata["intent"]
        
        # Keep last 10 turns
        if len(conversation["history"]) > 20:
            conversation["history"] = conversation["history"][-20:]
        
        if self.use_dynamodb:
            self.table.put_item(Item=conversation)
        else:
            self._local_store[conversation_id] = conversation
        
        return True
    
    def get_history(self, conversation_id: str, max_turns: int = 5) -> List[Dict]:
        """Get conversation history."""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
        history = conversation.get("history", [])
        return history[-max_turns * 2:]

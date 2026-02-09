"""Conversation state management."""

import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from .config import CONVERSATION_TIMEOUT_MINUTES, MAX_HISTORY_TURNS


class ConversationManager:
    """Manage conversation state and history."""
    
    def __init__(
        self,
        table_name: str = None,
        use_dynamodb: bool = False,
    ):
        """Initialize conversation manager.
        
        Args:
            table_name: DynamoDB table name (if using DynamoDB)
            use_dynamodb: Whether to use DynamoDB for persistence
        """
        self.use_dynamodb = use_dynamodb
        self.table_name = table_name
        
        # In-memory store for local testing
        self._local_store: Dict[str, Dict] = {}
        
        if use_dynamodb:
            self.dynamodb = boto3.resource("dynamodb")
            self.table = self.dynamodb.Table(table_name)
        else:
            self.dynamodb = None
            self.table = None
    
    def create_conversation(self, user_id: str = None) -> str:
        """Create a new conversation.
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            Conversation ID
        """
        conversation_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        conversation = {
            "conversation_id": conversation_id,
            "user_id": user_id or "anonymous",
            "created_at": timestamp,
            "updated_at": timestamp,
            "status": "active",
            "intent": None,
            "turn_count": 0,
            "history": [],
        }
        
        if self.use_dynamodb:
            self.table.put_item(Item=conversation)
        else:
            self._local_store[conversation_id] = conversation
        
        return conversation_id
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get conversation by ID.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation data or None if not found
        """
        if self.use_dynamodb:
            try:
                response = self.table.get_item(
                    Key={"conversation_id": conversation_id}
                )
                return response.get("Item")
            except ClientError:
                return None
        else:
            return self._local_store.get(conversation_id)
    
    def add_turn(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Dict = None,
    ) -> bool:
        """Add a conversation turn.
        
        Args:
            conversation_id: Conversation ID
            role: "user" or "assistant"
            content: Message content
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False
        
        # Check for timeout
        if self._is_timed_out(conversation):
            conversation["status"] = "timed_out"
            self._save_conversation(conversation)
            return False
        
        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        
        conversation["history"].append(turn)
        conversation["turn_count"] = len(conversation["history"]) // 2
        conversation["updated_at"] = datetime.utcnow().isoformat()
        
        # Update intent if provided in metadata
        if metadata and "intent" in metadata:
            conversation["intent"] = metadata["intent"]
        
        # Trim history if too long (keep last N turns)
        max_messages = MAX_HISTORY_TURNS * 2
        if len(conversation["history"]) > max_messages:
            conversation["history"] = conversation["history"][-max_messages:]
        
        self._save_conversation(conversation)
        return True
    
    def get_history(
        self,
        conversation_id: str,
        max_turns: int = MAX_HISTORY_TURNS,
    ) -> List[Dict]:
        """Get conversation history.
        
        Args:
            conversation_id: Conversation ID
            max_turns: Maximum number of turns to return
            
        Returns:
            List of conversation turns
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
        
        history = conversation.get("history", [])
        max_messages = max_turns * 2
        return history[-max_messages:]
    
    def end_conversation(self, conversation_id: str, reason: str = "completed") -> bool:
        """Mark conversation as ended.
        
        Args:
            conversation_id: Conversation ID
            reason: Reason for ending (completed, timed_out, escalated)
            
        Returns:
            True if successful
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False
        
        conversation["status"] = reason
        conversation["ended_at"] = datetime.utcnow().isoformat()
        
        self._save_conversation(conversation)
        return True
    
    def _is_timed_out(self, conversation: Dict) -> bool:
        """Check if conversation has timed out."""
        last_updated = datetime.fromisoformat(conversation["updated_at"])
        timeout = timedelta(minutes=CONVERSATION_TIMEOUT_MINUTES)
        return datetime.utcnow() - last_updated > timeout
    
    def _save_conversation(self, conversation: Dict):
        """Save conversation to storage."""
        if self.use_dynamodb:
            self.table.put_item(Item=conversation)
        else:
            self._local_store[conversation["conversation_id"]] = conversation
    
    def get_conversation_summary(self, conversation_id: str) -> Dict:
        """Get a summary of the conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation summary
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return {}
        
        history = conversation.get("history", [])
        user_messages = [t for t in history if t["role"] == "user"]
        assistant_messages = [t for t in history if t["role"] == "assistant"]
        
        return {
            "conversation_id": conversation_id,
            "user_id": conversation.get("user_id"),
            "status": conversation.get("status"),
            "intent": conversation.get("intent"),
            "turn_count": conversation.get("turn_count", 0),
            "duration_minutes": self._calculate_duration(conversation),
            "user_message_count": len(user_messages),
            "assistant_message_count": len(assistant_messages),
            "started_at": conversation.get("created_at"),
            "last_activity": conversation.get("updated_at"),
        }
    
    def _calculate_duration(self, conversation: Dict) -> int:
        """Calculate conversation duration in minutes."""
        try:
            created = datetime.fromisoformat(conversation["created_at"])
            updated = datetime.fromisoformat(conversation["updated_at"])
            return int((updated - created).total_seconds() / 60)
        except:
            return 0
    
    def list_active_conversations(self, user_id: str = None) -> List[Dict]:
        """List active conversations.
        
        Args:
            user_id: Filter by user ID
            
        Returns:
            List of conversation summaries
        """
        if self.use_dynamodb:
            if user_id:
                # Query by user_id (would need GSI in production)
                response = self.table.scan(
                    FilterExpression="#s = :status AND user_id = :user_id",
                    ExpressionAttributeNames={"#s": "status"},
                    ExpressionAttributeValues={
                        ":status": "active",
                        ":user_id": user_id,
                    }
                )
            else:
                response = self.table.scan(
                    FilterExpression="#s = :status",
                    ExpressionAttributeNames={"#s": "status"},
                    ExpressionAttributeValues={":status": "active"},
                )
            conversations = response.get("Items", [])
        else:
            conversations = [
                conv for conv in self._local_store.values()
                if conv.get("status") == "active"
            ]
            if user_id:
                conversations = [
                    conv for conv in conversations
                    if conv.get("user_id") == user_id
                ]
        
        return [self.get_conversation_summary(c["conversation_id"]) for c in conversations]

"""RAG (Retrieval-Augmented Generation) Engine."""

import logging
import os
from typing import Dict, List, Optional

import litellm

from .config import (
    DEFAULT_LLM_MODEL,
    FALLBACK_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
)
from .conversation_manager import ConversationManager
from .prompt_manager import PromptManager
from .vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure LiteLLM
litellm.set_verbose = False


class RAGEngine:
    """RAG engine for generating customer service responses."""
    
    def __init__(
        self,
        vector_store: VectorStore = None,
        conversation_manager: ConversationManager = None,
        prompt_manager: PromptManager = None,
        primary_model: str = DEFAULT_LLM_MODEL,
        fallback_model: str = FALLBACK_MODEL,
    ):
        """Initialize RAG engine.
        
        Args:
            vector_store: Vector store for retrieval
            conversation_manager: Conversation state manager
            prompt_manager: Prompt builder
            primary_model: Primary LLM model
            fallback_model: Fallback LLM model
        """
        self.vector_store = vector_store or VectorStore()
        self.conversation_manager = conversation_manager or ConversationManager()
        self.prompt_manager = prompt_manager or PromptManager()
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        
        # A/B testing state
        self.prompt_variants: Dict[str, str] = {}  # conversation_id -> variant
    
    def generate_response(
        self,
        query: str,
        conversation_id: str = None,
        intent: str = "general_inquiry",
        user_id: str = None,
        use_context: bool = True,
        use_history: bool = True,
        metadata: Dict = None,
    ) -> Dict:
        """Generate a response using RAG.
        
        Args:
            query: User query
            conversation_id: Existing conversation ID (creates new if None)
            intent: Classified intent
            user_id: User identifier
            use_context: Whether to use retrieved context
            use_history: Whether to use conversation history
            metadata: Additional metadata
            
        Returns:
            Response dictionary with content and metadata
        """
        # Create or get conversation
        if not conversation_id:
            conversation_id = self.conversation_manager.create_conversation(user_id)
        
        conversation = self.conversation_manager.get_conversation(conversation_id)
        if not conversation:
            conversation_id = self.conversation_manager.create_conversation(user_id)
        
        # Get conversation history
        history = []
        if use_history:
            history = self.conversation_manager.get_history(conversation_id)
        
        # Retrieve context from vector store
        retrieved_context = []
        if use_context:
            retrieved_context = self.vector_store.search(
                query=query,
                filter_dict={"intent": intent} if intent != "general_inquiry" else None,
            )
        
        # Get A/B test variant
        variant = self._get_prompt_variant(conversation_id)
        
        # Build prompt
        prompt = self.prompt_manager.build_prompt(
            query=query,
            intent=intent,
            retrieved_context=retrieved_context,
            conversation_history=history,
        )
        
        # Generate response
        try:
            response_text = self._call_llm(prompt, variant)
            model_used = self.primary_model
        except Exception as e:
            logger.error(f"Primary model failed: {e}")
            try:
                response_text = self._call_llm(prompt, variant, use_fallback=True)
                model_used = self.fallback_model
            except Exception as e2:
                logger.error(f"Fallback model also failed: {e2}")
                response_text = self._get_fallback_response(intent)
                model_used = "fallback_template"
        
        # Add user turn to conversation
        self.conversation_manager.add_turn(
            conversation_id=conversation_id,
            role="user",
            content=query,
            metadata={"intent": intent, **(metadata or {})},
        )
        
        # Add assistant turn to conversation
        self.conversation_manager.add_turn(
            conversation_id=conversation_id,
            role="assistant",
            content=response_text,
            metadata={"model": model_used, "variant": variant},
        )
        
        # Prepare response
        response = {
            "conversation_id": conversation_id,
            "response": response_text,
            "intent": intent,
            "model_used": model_used,
            "prompt_variant": variant,
            "retrieved_context": [
                {
                    "document": ctx["document"][:200] + "..." if len(ctx["document"]) > 200 else ctx["document"],
                    "score": round(ctx["score"], 3),
                }
                for ctx in retrieved_context[:3]
            ] if retrieved_context else [],
            "turn_number": len(history) // 2 + 1,
        }
        
        return response
    
    def _call_llm(
        self,
        prompt: str,
        variant: str = "A",
        use_fallback: bool = False,
    ) -> str:
        """Call LLM via LiteLLM.
        
        Args:
            prompt: Complete prompt
            variant: A/B test variant
            use_fallback: Whether to use fallback model
            
        Returns:
            Generated text
        """
        model = self.fallback_model if use_fallback else self.primary_model
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        response = litellm.completion(
            model=model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        
        return response.choices[0].message.content.strip()
    
    def _get_prompt_variant(self, conversation_id: str) -> str:
        """Get A/B test variant for conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Variant "A" or "B"
        """
        if conversation_id not in self.prompt_variants:
            # Simple hash-based assignment for consistency
            import hashlib
            hash_val = int(hashlib.md5(conversation_id.encode()).hexdigest(), 16)
            self.prompt_variants[conversation_id] = "A" if hash_val % 2 == 0 else "B"
        return self.prompt_variants[conversation_id]
    
    def _get_fallback_response(self, intent: str) -> str:
        """Get template fallback response."""
        fallbacks = {
            "return": "I understand you'd like to process a return. I'll connect you with a representative who can help you with this right away.",
            "grievance": "I sincerely apologize for this experience. I'm escalating this to a supervisor who will contact you within 24 hours.",
            "billing": "I understand you have a billing concern. Let me connect you with our billing department for immediate assistance.",
            "technical": "I see you're experiencing a technical issue. I'm creating a support ticket and our technical team will assist you shortly.",
            "support": "I'd be happy to help with your question. Let me find the most accurate information for you.",
            "escalation": "This is being treated as a priority case. A senior representative will contact you within the hour.",
        }
        return fallbacks.get(
            intent,
            "Thank you for contacting us. How can I assist you today?"
        )
    
    def add_feedback(
        self,
        conversation_id: str,
        turn_index: int,
        feedback: str,  # "positive" or "negative"
        feedback_text: str = None,
    ) -> bool:
        """Add feedback for a response.
        
        Args:
            conversation_id: Conversation ID
            turn_index: Index of the turn (1-based)
            feedback: "positive" or "negative"
            feedback_text: Optional detailed feedback
            
        Returns:
            True if successful
        """
        conversation = self.conversation_manager.get_conversation(conversation_id)
        if not conversation:
            return False
        
        history = conversation.get("history", [])
        assistant_turns = [i for i, t in enumerate(history) if t["role"] == "assistant"]
        
        if turn_index < 1 or turn_index > len(assistant_turns):
            return False
        
        turn_idx = assistant_turns[turn_index - 1]
        
        if "feedback" not in history[turn_idx]["metadata"]:
            history[turn_idx]["metadata"]["feedback"] = {}
        
        history[turn_idx]["metadata"]["feedback"] = {
            "type": feedback,
            "text": feedback_text,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        self.conversation_manager._save_conversation(conversation)
        return True
    
    def get_conversation_context(self, conversation_id: str) -> Dict:
        """Get full conversation context."""
        return self.conversation_manager.get_conversation(conversation_id)


# Import for type hint
from datetime import datetime

"""Prompt management for Lambda."""

from typing import Dict, List, Optional

MAX_HISTORY_TURNS = 3

INTENT_PROMPTS = {
    "return": "You are a helpful customer service assistant specializing in returns and refunds.",
    "grievance": "You are a senior customer service representative handling escalated complaints.",
    "billing": "You are a billing specialist helping with payment and refund issues.",
    "technical": "You are a technical support specialist helping customers with product issues.",
    "support": "You are a product support specialist providing helpful information.",
    "general_inquiry": "You are a friendly customer service representative.",
    "escalation": "You are handling a priority case requiring immediate attention.",
}


class PromptManager:
    """Manage prompts for different intents."""
    
    def __init__(self, max_history: int = MAX_HISTORY_TURNS):
        self.max_history = max_history
    
    def build_prompt(
        self,
        query: str,
        intent: str,
        retrieved_context: Optional[List[Dict]] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        """Build a complete prompt."""
        system_prompt = INTENT_PROMPTS.get(intent, INTENT_PROMPTS["general_inquiry"])
        
        # Build history string
        history_str = ""
        if conversation_history:
            recent = conversation_history[-self.max_history * 2:]
            for turn in recent:
                role = "Customer" if turn["role"] == "user" else "Assistant"
                history_str += f"{role}: {turn['content']}\n"
        
        # Build context string
        context_str = ""
        if retrieved_context:
            context_str = "\n".join([
                f"{i+1}. {ctx.get('document', '')[:200]}"
                for i, ctx in enumerate(retrieved_context[:3])
            ])
        
        prompt = f"""{system_prompt}

Context:
{context_str}

Conversation History:
{history_str}

Customer: {query}

Assistant:"""
        
        return prompt

"""Prompt management for different intents and conversation contexts."""

from typing import Dict, List, Optional

from .config import (
    DEFAULT_SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
    INTENT_SYSTEM_PROMPTS,
    MAX_HISTORY_TURNS,
)


class PromptManager:
    """Manage prompts for different intents and conversation contexts."""
    
    def __init__(
        self,
        max_history: int = MAX_HISTORY_TURNS,
    ):
        """Initialize prompt manager.
        
        Args:
            max_history: Maximum number of conversation turns to include
        """
        self.max_history = max_history
    
    def build_prompt(
        self,
        query: str,
        intent: str,
        retrieved_context: Optional[List[Dict]] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        """Build a complete prompt for the LLM.
        
        Args:
            query: Current user query
            intent: Classified intent
            retrieved_context: Retrieved FAQ documents
            conversation_history: Previous conversation turns
            
        Returns:
            Complete prompt string
        """
        # System prompt
        system_prompt = INTENT_SYSTEM_PROMPTS.get(intent, DEFAULT_SYSTEM_PROMPT)
        
        # Build context sections
        context_section = self._build_context_section(retrieved_context)
        history_section = self._build_history_section(conversation_history)
        few_shot_section = self._build_few_shot_section(intent)
        
        # Combine all sections
        prompt_parts = [
            f"System: {system_prompt}",
            "",
        ]
        
        if context_section:
            prompt_parts.extend([
                "Context from FAQ:",
                context_section,
                "",
            ])
        
        if few_shot_section:
            prompt_parts.extend([
                "Examples of good responses:",
                few_shot_section,
                "",
            ])
        
        if history_section:
            prompt_parts.extend([
                "Conversation History:",
                history_section,
                "",
            ])
        
        prompt_parts.extend([
            f"Customer: {query}",
            "Assistant:",
        ])
        
        return "\n".join(prompt_parts)
    
    def _build_context_section(
        self,
        retrieved_context: Optional[List[Dict]]
    ) -> str:
        """Build context section from retrieved documents."""
        if not retrieved_context:
            return ""
        
        context_parts = []
        for i, doc in enumerate(retrieved_context[:3], 1):  # Top 3 docs
            text = doc.get("document", "")
            # Extract just the answer if it's a Q&A format
            if "A:" in text:
                text = text.split("A:", 1)[1].strip()
            context_parts.append(f"{i}. {text}")
        
        return "\n".join(context_parts)
    
    def _build_history_section(
        self,
        conversation_history: Optional[List[Dict]]
    ) -> str:
        """Build conversation history section."""
        if not conversation_history:
            return ""
        
        # Take last N turns
        recent_history = conversation_history[-self.max_history * 2:]
        
        history_parts = []
        for turn in recent_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                history_parts.append(f"Customer: {content}")
            else:
                history_parts.append(f"Assistant: {content}")
        
        return "\n".join(history_parts)
    
    def _build_few_shot_section(self, intent: str) -> str:
        """Build few-shot examples section."""
        examples = FEW_SHOT_EXAMPLES.get(intent, [])
        if not examples:
            return ""
        
        example_parts = []
        for ex in examples[:2]:  # Top 2 examples
            example_parts.append(f"Customer: {ex['user']}")
            example_parts.append(f"Assistant: {ex['assistant']}")
            example_parts.append("")
        
        return "\n".join(example_parts).strip()
    
    def build_follow_up_prompt(
        self,
        query: str,
        intent: str,
        previous_response: str,
        conversation_history: List[Dict],
    ) -> str:
        """Build prompt for follow-up responses.
        
        Args:
            query: Current user query
            intent: Classified intent
            previous_response: Previous assistant response
            conversation_history: Full conversation history
            
        Returns:
            Complete prompt string
        """
        system_prompt = INTENT_SYSTEM_PROMPTS.get(intent, DEFAULT_SYSTEM_PROMPT)
        system_prompt += "\nThis is a follow-up question. Maintain context from the previous conversation."
        
        history_section = self._build_history_section(conversation_history)
        
        prompt = f"""System: {system_prompt}

Previous Response:
{previous_response}

Conversation History:
{history_section}

Customer Follow-up: {query}

Assistant:"""
        
        return prompt
    
    def get_prompt_variant(self, intent: str, variant: str = "A") -> str:
        """Get A/B testing prompt variant.
        
        Args:
            intent: Intent category
            variant: "A" or "B" for A/B testing
            
        Returns:
            Modified system prompt
        """
        base_prompt = INTENT_SYSTEM_PROMPTS.get(intent, DEFAULT_SYSTEM_PROMPT)
        
        if variant == "B":
            # Variant B: More empathetic and proactive
            if intent == "grievance":
                return base_prompt + "\nAcknowledge the emotional impact of the situation explicitly."
            elif intent == "technical":
                return base_prompt + "\nOffer to schedule a callback with a specialist proactively."
            else:
                return base_prompt + "\nBe more conversational and ask clarifying questions."
        
        return base_prompt

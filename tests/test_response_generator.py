"""Tests for response generation module."""

import pytest
import sys
sys.path.insert(0, "src")

from src.response_generator.conversation_manager import ConversationManager
from src.response_generator.prompt_manager import PromptManager
from src.response_generator.vector_store import VectorStore


class TestConversationManager:
    """Test conversation management."""
    
    @pytest.fixture
    def manager(self):
        """Create conversation manager."""
        return ConversationManager(use_dynamodb=False)
    
    def test_create_conversation(self, manager):
        """Test creating a conversation."""
        conv_id = manager.create_conversation(user_id="test_user")
        assert conv_id is not None
        assert len(conv_id) > 0
        
        # Verify it exists
        conv = manager.get_conversation(conv_id)
        assert conv is not None
        assert conv["user_id"] == "test_user"
        assert conv["status"] == "active"
    
    def test_add_turn(self, manager):
        """Test adding conversation turns."""
        conv_id = manager.create_conversation()
        
        # Add user turn
        success = manager.add_turn(conv_id, "user", "Hello", {"intent": "greeting"})
        assert success
        
        # Add assistant turn
        success = manager.add_turn(conv_id, "assistant", "Hi there!")
        assert success
        
        # Check history
        history = manager.get_history(conv_id)
        assert len(history) == 2
    
    def test_history_limit(self, manager):
        """Test history is limited."""
        conv_id = manager.create_conversation()
        
        # Add many turns
        for i in range(20):
            manager.add_turn(conv_id, "user", f"Message {i}")
            manager.add_turn(conv_id, "assistant", f"Response {i}")
        
        # History should be limited
        history = manager.get_history(conv_id, max_turns=5)
        assert len(history) <= 10  # 5 turns * 2 messages


class TestPromptManager:
    """Test prompt building."""
    
    @pytest.fixture
    def prompt_manager(self):
        """Create prompt manager."""
        return PromptManager()
    
    def test_build_prompt(self, prompt_manager):
        """Test prompt building."""
        prompt = prompt_manager.build_prompt(
            query="I want to return my order",
            intent="return",
        )
        assert "return" in prompt
        assert "I want to return my order" in prompt
    
    def test_build_prompt_with_context(self, prompt_manager):
        """Test prompt with retrieved context."""
        context = [
            {"document": "Q: Return policy?\nA: 30 days"},
        ]
        prompt = prompt_manager.build_prompt(
            query="Can I return this?",
            intent="return",
            retrieved_context=context,
        )
        assert "30 days" in prompt


class TestVectorStore:
    """Test vector store operations."""
    
    @pytest.fixture
    def vector_store(self, tmp_path):
        """Create temporary vector store."""
        return VectorStore(persist_dir=str(tmp_path / "chroma"))
    
    def test_add_and_search(self, vector_store):
        """Test adding and searching documents."""
        # Add documents
        docs = [
            "Returns can be made within 30 days",
            "Refunds process in 5-7 business days",
        ]
        ids = vector_store.add_documents(docs)
        assert len(ids) == 2
        
        # Search
        results = vector_store.search("How do I return an item?", top_k=1)
        assert len(results) > 0
        assert "return" in results[0]["document"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

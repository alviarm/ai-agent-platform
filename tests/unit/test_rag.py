"""
Unit Tests for RAG Response Engine

Tests:
- Retrieval accuracy: top-3 chunks should contain answer for 80% of queries
- Prompt templates render correctly with context
- Mock LLM responses (deterministic echo or small model)
- Conversation history truncation (max 3 previous turns)
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import RAG components
from src.response_generator.rag_engine import RAGEngine
from src.response_generator.vector_store import VectorStore
from src.response_generator.conversation_manager import ConversationManager
from src.response_generator.prompt_manager import PromptManager


# Test configuration
TEST_CHROMA_PATH = Path("data/chroma")
MIN_RETRIEVAL_ACCURACY = 0.80
MAX_HISTORY_TURNS = 3
RETRIEVAL_TEST_QUERIES = [
    {
        "query": "How do I return my laptop?",
        "expected_keywords": ["return", "refund", "days", "unused"],
        "intent": "return",
    },
    {
        "query": "I was charged twice on my credit card",
        "expected_keywords": ["charge", "billing", "duplicate", "refund"],
        "intent": "billing",
    },
    {
        "query": "The app keeps crashing when I try to login",
        "expected_keywords": ["app", "crash", "login", "troubleshoot"],
        "intent": "technical",
    },
    {
        "query": "Is this product compatible with iPhone?",
        "expected_keywords": ["compatible", "product", "device"],
        "intent": "support",
    },
    {
        "query": "How long does shipping take to Canada?",
        "expected_keywords": ["shipping", "days", "delivery"],
        "intent": "general_inquiry",
    },
]


@pytest.fixture(scope="module")
def vector_store():
    """Initialize vector store with test data."""
    persist_dir = str(TEST_CHROMA_PATH) if TEST_CHROMA_PATH.exists() else "./data/chroma"
    
    try:
        store = VectorStore(persist_dir=persist_dir)
        # Ensure we have documents
        if store.collection.count() == 0:
            pytest.skip("Vector store is empty - run seed_chroma.py first")
        return store
    except Exception as e:
        pytest.skip(f"Could not initialize vector store: {e}")


@pytest.fixture
def conversation_manager():
    """Initialize conversation manager."""
    return ConversationManager(use_dynamodb=False)


@pytest.fixture
def prompt_manager():
    """Initialize prompt manager."""
    return PromptManager()


@pytest.fixture
def rag_engine(vector_store, conversation_manager, prompt_manager):
    """Initialize RAG engine with mocked LLM."""
    engine = RAGEngine(
        vector_store=vector_store,
        conversation_manager=conversation_manager,
        prompt_manager=prompt_manager,
    )
    return engine


class TestRetrievalAccuracy:
    """Tests for document retrieval accuracy."""
    
    def test_retrieval_returns_results(self, vector_store):
        """Test that retrieval returns results for queries."""
        query = "How do I return an item?"
        results = vector_store.search(query, top_k=3)
        
        assert len(results) > 0, "No documents retrieved"
        assert all("document" in r for r in results), "Results missing document field"
        assert all("score" in r for r in results), "Results missing score field"
        
        # Scores should be between 0 and 1
        for r in results:
            assert 0 <= r["score"] <= 1, f"Invalid score: {r['score']}"
    
    def test_retrieval_relevance(self, vector_store):
        """Test that retrieved documents are relevant to query."""
        successful_queries = 0
        
        for test_case in RETRIEVAL_TEST_QUERIES:
            query = test_case["query"]
            expected_keywords = test_case["expected_keywords"]
            
            results = vector_store.search(query, top_k=3)
            
            # Check if any of the top results contain expected keywords
            found_relevant = False
            for result in results:
                doc_text = result["document"].lower()
                # Check if at least 2 expected keywords appear
                keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in doc_text)
                if keyword_hits >= 2:
                    found_relevant = True
                    break
            
            if found_relevant:
                successful_queries += 1
            
            print(f"\nQuery: {query}")
            print(f"  Top result: {results[0]['document'][:100]}...")
            print(f"  Score: {results[0]['score']:.3f}")
            print(f"  Relevant: {found_relevant}")
        
        accuracy = successful_queries / len(RETRIEVAL_TEST_QUERIES)
        print(f"\nRetrieval Accuracy: {accuracy:.1%} ({successful_queries}/{len(RETRIEVAL_TEST_QUERIES)})")
        
        assert accuracy >= MIN_RETRIEVAL_ACCURACY, (
            f"Retrieval accuracy {accuracy:.1%} below threshold {MIN_RETRIEVAL_ACCURACY:.1%}"
        )
    
    def test_retrieval_with_intent_filter(self, vector_store):
        """Test retrieval with intent-based filtering."""
        query = "How do I return something?"
        
        # Search without filter
        results_no_filter = vector_store.search(query, top_k=5)
        
        # Search with return filter
        results_filtered = vector_store.search(
            query, 
            top_k=5,
            filter_dict={"intent": "return"}
        )
        
        # Filtered results should all have return intent
        for r in results_filtered:
            assert r["metadata"].get("intent") == "return", (
                f"Filtered result has wrong intent: {r['metadata']}"
            )
        
        print(f"\nNo filter: {len(results_no_filter)} results")
        print(f"With filter: {len(results_filtered)} results")
    
    def test_retrieval_scores_ranking(self, vector_store):
        """Test that retrieval scores are properly ranked."""
        query = "Return policy question"
        results = vector_store.search(query, top_k=5)
        
        # Scores should be in descending order
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Results not properly ranked by score"
        
        print(f"\nRetrieval scores (descending): {scores}")


class TestPromptTemplates:
    """Tests for prompt template rendering."""
    
    def test_prompt_building_with_context(self, prompt_manager):
        """Test that prompts are built correctly with context."""
        query = "How do I return an item?"
        intent = "return"
        retrieved_context = [
            {"document": "Returns accepted within 30 days with receipt."},
            {"document": "Items must be in original packaging."},
        ]
        
        prompt = prompt_manager.build_prompt(
            query=query,
            intent=intent,
            retrieved_context=retrieved_context,
            conversation_history=[],
        )
        
        # Prompt should contain all components
        assert query in prompt, "Query not in prompt"
        assert any(ctx["document"] in prompt for ctx in retrieved_context), (
            "Context not in prompt"
        )
        
        print(f"\nGenerated prompt:\n{prompt[:500]}...")
    
    def test_prompt_with_conversation_history(self, prompt_manager):
        """Test prompt building with conversation history."""
        history = [
            {"role": "user", "content": "I want to return something"},
            {"role": "assistant", "content": "I can help with returns"},
            {"role": "user", "content": "How long do I have?"},
        ]
        
        prompt = prompt_manager.build_prompt(
            query="What about without receipt?",
            intent="return",
            retrieved_context=[{"document": "Returns need receipt."}],
            conversation_history=history,
        )
        
        # History should be included
        assert "I want to return something" in prompt or "user:" in prompt.lower(), (
            "History not in prompt"
        )
        
        print(f"\nPrompt with history:\n{prompt[:500]}...")
    
    def test_prompt_without_context(self, prompt_manager):
        """Test prompt building without retrieved context."""
        prompt = prompt_manager.build_prompt(
            query="Hello",
            intent="general_inquiry",
            retrieved_context=[],
            conversation_history=[],
        )
        
        assert "Hello" in prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestConversationHistory:
    """Tests for conversation history management."""
    
    def test_history_truncation(self, conversation_manager):
        """Test that conversation history is properly truncated."""
        # Create conversation
        conv_id = conversation_manager.create_conversation(user_id="test_user")
        
        # Add more turns than the limit
        for i in range(10):
            conversation_manager.add_turn(
                conversation_id=conv_id,
                role="user",
                content=f"User message {i}",
            )
            conversation_manager.add_turn(
                conversation_id=conv_id,
                role="assistant",
                content=f"Assistant response {i}",
            )
        
        # Get history
        history = conversation_manager.get_history(conv_id, max_turns=MAX_HISTORY_TURNS)
        
        # Should be limited to max_turns
        assert len(history) <= MAX_HISTORY_TURNS * 2, (
            f"History not truncated: {len(history)} messages"
        )
        
        # Should keep most recent turns
        assert "User message 9" in [h["content"] for h in history], (
            "Recent messages not preserved"
        )
        
        print(f"\nHistory length: {len(history)} messages (max: {MAX_HISTORY_TURNS * 2})")
    
    def test_conversation_creation_and_retrieval(self, conversation_manager):
        """Test conversation creation and retrieval."""
        conv_id = conversation_manager.create_conversation(user_id="test_user")
        
        assert conv_id is not None
        assert isinstance(conv_id, str)
        
        # Retrieve conversation
        conversation = conversation_manager.get_conversation(conv_id)
        assert conversation is not None
        assert conversation["conversation_id"] == conv_id
        assert conversation["user_id"] == "test_user"
        assert conversation["status"] == "active"
    
    def test_add_turn_to_conversation(self, conversation_manager):
        """Test adding turns to a conversation."""
        conv_id = conversation_manager.create_conversation()
        
        # Add user turn
        success = conversation_manager.add_turn(
            conversation_id=conv_id,
            role="user",
            content="Test message",
            metadata={"intent": "test"},
        )
        assert success
        
        # Verify turn was added
        conversation = conversation_manager.get_conversation(conv_id)
        assert len(conversation["history"]) == 1
        assert conversation["history"][0]["content"] == "Test message"
        assert conversation["history"][0]["metadata"]["intent"] == "test"


class TestRAGEngineIntegration:
    """Integration tests for the RAG engine."""
    
    def test_generate_response_structure(self, rag_engine):
        """Test that generate_response returns expected structure."""
        # Mock the LLM call to avoid external dependencies
        with patch.object(rag_engine, '_call_llm', return_value="Mock response"):
            result = rag_engine.generate_response(
                query="How do I return an item?",
                intent="return",
            )
        
        # Check response structure
        assert "conversation_id" in result
        assert "response" in result
        assert "intent" in result
        assert "retrieved_context" in result
        assert "turn_number" in result
        
        assert result["response"] == "Mock response"
        assert result["intent"] == "return"
        assert isinstance(result["retrieved_context"], list)
    
    def test_conversation_state_persistence(self, rag_engine):
        """Test that conversation state is persisted across calls."""
        with patch.object(rag_engine, '_call_llm', return_value="Response"):
            # First call
            result1 = rag_engine.generate_response(
                query="Hello",
                intent="general_inquiry",
            )
            conv_id = result1["conversation_id"]
            
            # Second call with same conversation
            result2 = rag_engine.generate_response(
                query="I have a question",
                conversation_id=conv_id,
                intent="general_inquiry",
            )
            
            assert result2["conversation_id"] == conv_id
            assert result2["turn_number"] == 2
    
    def test_rag_with_mock_llm(self, rag_engine):
        """Test RAG pipeline with mocked LLM."""
        query = "What is your return policy?"
        
        # Capture the prompt that would be sent to LLM
        captured_prompts = []
        
        def mock_llm(prompt, variant="A", use_fallback=False):
            captured_prompts.append(prompt)
            return f"Response to: {query}"
        
        with patch.object(rag_engine, '_call_llm', side_effect=mock_llm):
            result = rag_engine.generate_response(
                query=query,
                intent="return",
            )
        
        # Verify prompt was built and sent
        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        
        # Prompt should contain query and retrieved context
        assert query in prompt
        assert len(result["retrieved_context"]) > 0
        
        print(f"\nCaptured prompt:\n{prompt[:800]}...")
        print(f"\nRetrieved {len(result['retrieved_context'])} context chunks")
    
    def test_fallback_response_on_llm_failure(self, rag_engine):
        """Test that fallback responses work when LLM fails."""
        with patch.object(rag_engine, '_call_llm', side_effect=Exception("LLM Error")):
            result = rag_engine.generate_response(
                query="Help with return",
                intent="return",
            )
        
        # Should still return a response (fallback)
        assert result["response"] is not None
        assert len(result["response"]) > 0
        assert result["model_used"] == "fallback_template"
    
    def test_retrieved_context_in_response(self, rag_engine):
        """Test that retrieved context is included in response."""
        with patch.object(rag_engine, '_call_llm', return_value="Response"):
            result = rag_engine.generate_response(
                query="How do I return my laptop?",
                intent="return",
            )
        
        # Should have retrieved context
        assert len(result["retrieved_context"]) > 0
        
        # Context should have required fields
        for ctx in result["retrieved_context"]:
            assert "document" in ctx
            assert "score" in ctx
            assert isinstance(ctx["score"], float)


class TestVectorStoreOperations:
    """Tests for vector store operations."""
    
    def test_add_and_retrieve_document(self, vector_store):
        """Test adding and retrieving a document."""
        doc_text = "Test document for unit testing"
        doc_metadata = {"category": "test", "intent": "general_inquiry"}
        
        # Add document
        doc_ids = vector_store.add_documents(
            documents=[doc_text],
            metadatas=[doc_metadata],
        )
        
        assert len(doc_ids) == 1
        
        # Search for it
        results = vector_store.search("test document", top_k=5)
        
        # Should find our document
        found = any(doc_text in r["document"] for r in results)
        assert found, "Added document not found in search"
    
    def test_get_all_documents(self, vector_store):
        """Test retrieving all documents."""
        docs = vector_store.get_all_documents()
        
        assert isinstance(docs, list)
        assert len(docs) > 0
        
        # Each doc should have required fields
        for doc in docs:
            assert "id" in doc
            assert "document" in doc
            assert "metadata" in doc
    
    def test_document_chunking(self, vector_store):
        """Test document chunking functionality."""
        long_text = " ".join(["word"] * 1000)  # Long text
        
        chunks = vector_store.chunk_document(long_text, chunk_size=100, chunk_overlap=20)
        
        assert len(chunks) > 1, "Long document not properly chunked"
        
        # Each chunk should be reasonable size
        for chunk in chunks:
            words = chunk.split()
            assert len(words) <= 120  # Allow some margin


def test_end_to_end_rag_pipeline(vector_store, conversation_manager):
    """End-to-end test of the RAG pipeline."""
    print("\n" + "=" * 60)
    print("End-to-End RAG Pipeline Test")
    print("=" * 60)
    
    engine = RAGEngine(
        vector_store=vector_store,
        conversation_manager=conversation_manager,
    )
    
    # Test query
    query = "How do I return a damaged item?"
    
    # First, test retrieval
    retrieved = vector_store.search(query, top_k=3)
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(retrieved)} documents:")
    for i, r in enumerate(retrieved, 1):
        print(f"  {i}. [{r['metadata'].get('intent', 'unknown')}] Score: {r['score']:.3f}")
        print(f"     {r['document'][:150]}...")
    
    # Check relevance
    return_docs = [r for r in retrieved if r['metadata'].get('intent') == 'return']
    assert len(return_docs) >= 2, "Not enough return-related documents retrieved"
    
    # Mock LLM for full pipeline test
    with patch.object(engine, '_call_llm', return_value="To return a damaged item, contact us immediately with photos. We'll send a replacement."):
        result = engine.generate_response(query=query, intent="return")
    
    print(f"\nGenerated response: {result['response']}")
    print(f"Conversation ID: {result['conversation_id']}")
    print(f"Turn number: {result['turn_number']}")
    
    assert result["response"] is not None
    assert result["intent"] == "return"
    
    print("\n✓ End-to-end RAG pipeline test passed!")

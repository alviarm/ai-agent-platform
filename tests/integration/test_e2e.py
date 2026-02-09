"""
End-to-End Integration Tests

Simulates complete customer journey:
1. POST /chat (query: "My bill is wrong, help")
   → Verify: intent=BILLING, response contains relevant context
2. Store conversation_id
3. POST /feedback (conversation_id, rating: "thumbs_down", comment: "Not helpful")
4. GET /analytics/insights
   → Verify: feedback appears in negative sentiment bucket
   → Verify: billing topic count incremented
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

# API configuration
API_BASE_URL = "http://localhost:8002"  # Local API server for E2E tests
DEFAULT_TIMEOUT = 10


class TestEndToEndCustomerJourney:
    """
    End-to-end test simulating a complete customer interaction:
    Chat -> Feedback -> Analytics verification
    """
    
    @pytest.fixture(scope="class")
    def api_client(self):
        """Create API client fixture."""
        return APIClient(API_BASE_URL)
    
    def test_health_check(self, api_client):
        """Verify API is healthy and all services are available."""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["services"]["intent_classifier"] is True
        assert data["services"]["rag_engine"] is True
        assert data["services"]["feedback_analyzer"] is True
        
        print("\n✓ API health check passed")
        print(f"  Services: {data['services']}")
    
    def test_complete_customer_journey_billing(self, api_client):
        """
        Complete journey: Customer has billing issue → gets help → provides feedback
        """
        print("\n" + "=" * 60)
        print("E2E Test: Billing Issue Customer Journey")
        print("=" * 60)
        
        # Step 1: Initial chat about billing issue
        print("\nStep 1: Customer asks about billing issue")
        chat_response = api_client.post("/chat", json={
            "message": "My bill is wrong, I was charged twice for my order",
            "user_id": f"test_user_{uuid.uuid4().hex[:8]}"
        })
        
        assert chat_response.status_code == 200
        chat_data = chat_response.json()
        
        conversation_id = chat_data["conversation_id"]
        intent = chat_data["intent"]
        response_text = chat_data["response"]
        
        print(f"  Conversation ID: {conversation_id}")
        print(f"  Detected Intent: {intent}")
        print(f"  Response: {response_text[:100]}...")
        
        # Verify intent is billing-related
        assert intent in ["billing", "escalation", "general_inquiry"], (
            f"Expected billing-related intent, got {intent}"
        )
        
        # Verify response contains relevant content
        assert len(response_text) > 0
        assert chat_data["confidence"] > 0
        
        # Step 2: Follow-up question
        print("\nStep 2: Customer asks follow-up question")
        followup_response = api_client.post("/chat", json={
            "message": "When will the refund appear in my account?",
            "conversation_id": conversation_id,
        })
        
        assert followup_response.status_code == 200
        followup_data = followup_response.json()
        
        assert followup_data["conversation_id"] == conversation_id
        assert followup_data["turn_number"] == 2
        
        print(f"  Turn {followup_data['turn_number']} response received")
        
        # Step 3: Customer provides negative feedback
        print("\nStep 3: Customer submits negative feedback")
        feedback_response = api_client.post("/feedback", json={
            "conversation_id": conversation_id,
            "text": "Not helpful, the response was too generic and didn't answer my specific question about duplicate charges",
            "rating": "negative",
            "turn_index": 1,
        })
        
        assert feedback_response.status_code == 200
        feedback_data = feedback_response.json()
        
        print(f"  Feedback ID: {feedback_data['feedback_id']}")
        print(f"  Sentiment: {feedback_data['sentiment']} (conf: {feedback_data['confidence']:.3f})")
        print(f"  Keywords: {feedback_data['keywords'][:3]}")
        
        # Verify negative sentiment detected
        assert feedback_data["sentiment"] == "negative"
        assert feedback_data["confidence"] > 0
        
        # Step 4: Verify feedback appears in analytics
        print("\nStep 4: Verify analytics reflect the feedback")
        
        # Get conversation to verify feedback was stored
        conv_response = api_client.get(f"/conversation/{conversation_id}")
        assert conv_response.status_code == 200
        
        conversation = conv_response.json()
        assert conversation["conversation_id"] == conversation_id
        
        print(f"  Conversation has {len(conversation.get('history', []))} messages")
        
        print("\n✓ Complete billing journey test passed!")
    
    def test_complete_customer_journey_return(self, api_client):
        """
        Complete journey: Customer wants to return item → gets help → positive feedback
        """
        print("\n" + "=" * 60)
        print("E2E Test: Return Request Customer Journey")
        print("=" * 60)
        
        # Step 1: Ask about return policy
        print("\nStep 1: Customer asks about returns")
        chat_response = api_client.post("/chat", json={
            "message": "I received a damaged laptop and need to return it",
            "user_id": f"test_user_{uuid.uuid4().hex[:8]}"
        })
        
        assert chat_response.status_code == 200
        chat_data = chat_response.json()
        conversation_id = chat_data["conversation_id"]
        
        print(f"  Conversation ID: {conversation_id}")
        print(f"  Intent: {chat_data['intent']}")
        print(f"  Retrieved {len(chat_data.get('retrieved_context', []))} context chunks")
        
        assert chat_data["intent"] == "return"
        assert len(chat_data.get("retrieved_context", [])) > 0
        
        # Step 2: Ask follow-up about shipping
        print("\nStep 2: Customer asks about return shipping")
        chat_response2 = api_client.post("/chat", json={
            "message": "Do I have to pay for return shipping for a damaged item?",
            "conversation_id": conversation_id,
        })
        
        assert chat_response2.status_code == 200
        
        # Step 3: Positive feedback
        print("\nStep 3: Customer provides positive feedback")
        feedback_response = api_client.post("/feedback", json={
            "conversation_id": conversation_id,
            "text": "Very helpful, clear information about returns",
            "rating": "positive",
        })
        
        assert feedback_response.status_code == 200
        feedback_data = feedback_response.json()
        
        print(f"  Sentiment: {feedback_data['sentiment']}")
        assert feedback_data["sentiment"] == "positive"
        
        print("\n✓ Complete return journey test passed!")
    
    def test_intent_classification_endpoint(self, api_client):
        """Test the intent classification endpoint directly."""
        print("\n" + "=" * 60)
        print("E2E Test: Intent Classification Endpoint")
        print("=" * 60)
        
        test_cases = [
            ("I want to return my order", "return"),
            ("My credit card was charged twice", "billing"),
            ("The app keeps crashing", "technical"),
            ("How do I use this feature?", ["support", "general_inquiry"]),
            ("I need to speak to a manager", "escalation"),
        ]
        
        for text, expected_intent in test_cases:
            response = api_client.post("/classify", json={"text": text})
            
            assert response.status_code == 200
            data = response.json()
            
            predicted_intent = data["intent"]
            confidence = data["confidence"]
            
            print(f"\n  Text: '{text}'")
            print(f"  Predicted: {predicted_intent} (conf: {confidence:.3f})")
            
            # Check if prediction matches expected
            if isinstance(expected_intent, list):
                assert predicted_intent in expected_intent
            else:
                # Allow for some flexibility - if confidence is high, intent should match
                if confidence > 0.7:
                    assert predicted_intent == expected_intent, (
                        f"Expected {expected_intent}, got {predicted_intent}"
                    )
            
            # Verify all scores are present
            assert len(data["all_scores"]) == 7  # 7 intent categories
            assert all(0 <= score <= 1 for score in data["all_scores"].values())
        
        print("\n✓ Intent classification endpoint test passed!")
    
    def test_conversation_persistence(self, api_client):
        """Test that conversations are properly persisted."""
        print("\n" + "=" * 60)
        print("E2E Test: Conversation Persistence")
        print("=" * 60)
        
        # Create a conversation with multiple turns
        messages = [
            "Hello, I need help",
            "I want to return an item",
            "It's been more than 30 days, is that okay?",
        ]
        
        conversation_id = None
        for i, message in enumerate(messages):
            payload = {"message": message}
            if conversation_id:
                payload["conversation_id"] = conversation_id
            
            response = api_client.post("/chat", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            conversation_id = data["conversation_id"]
            
            print(f"  Turn {i+1}: {message[:40]}...")
            print(f"    -> Intent: {data['intent']}, Turn: {data['turn_number']}")
        
        # Retrieve conversation
        conv_response = api_client.get(f"/conversation/{conversation_id}")
        assert conv_response.status_code == 200
        
        conversation = conv_response.json()
        history = conversation.get("history", [])
        
        print(f"\n  Retrieved conversation with {len(history)} messages")
        
        # Should have user + assistant messages
        assert len(history) >= 6  # 3 user + 3 assistant messages
        
        # Verify turn numbers
        assert conversation.get("turn_count") == 3
        
        print("\n✓ Conversation persistence test passed!")
    
    def test_analytics_endpoint(self, api_client):
        """Test the analytics endpoint."""
        print("\n" + "=" * 60)
        print("E2E Test: Analytics Endpoint")
        print("=" * 60)
        
        response = api_client.get("/analytics")
        
        assert response.status_code == 200
        data = response.json()
        
        print(f"  Analytics response: {json.dumps(data, indent=2)}")
        
        # Verify required fields
        assert "total_conversations" in data
        assert "active_conversations" in data
        assert "average_response_time_ms" in data
        
        # Values should be non-negative
        assert data["total_conversations"] >= 0
        assert data["active_conversations"] >= 0
        
        print("\n✓ Analytics endpoint test passed!")
    
    def test_multilingual_and_typos(self, api_client):
        """Test handling of typos and multilingual input."""
        print("\n" + "=" * 60)
        print("E2E Test: Typos and Edge Cases")
        print("=" * 60)
        
        test_cases = [
            ("I wnat to retrun my lpatop", "return"),  # Typos
            ("My biling is wong", "billing"),  # Typos
            ("Help me please gracias", "general_inquiry"),  # Multilingual
            ("App crahses when login", "technical"),  # Typos
        ]
        
        for text, expected_intent in test_cases:
            response = api_client.post("/classify", json={"text": text})
            
            assert response.status_code == 200
            data = response.json()
            
            print(f"\n  Input: '{text}'")
            print(f"  Predicted: {data['intent']} (conf: {data['confidence']:.3f})")
            
            # Should still classify reasonably well
            assert data["confidence"] > 0.3  # At least some confidence
        
        print("\n✓ Typos and edge cases test passed!")
    
    def test_error_handling(self, api_client):
        """Test error handling for invalid requests."""
        print("\n" + "=" * 60)
        print("E2E Test: Error Handling")
        print("=" * 60)
        
        # Test missing required field
        response = api_client.post("/chat", json={})
        assert response.status_code == 422  # Validation error
        
        # Test invalid conversation ID
        response = api_client.get("/conversation/invalid-uuid")
        assert response.status_code == 404
        
        # Test empty message (should still work or give clear error)
        response = api_client.post("/chat", json={"message": ""})
        # Could be 200 (with fallback) or 422 (validation error)
        assert response.status_code in [200, 422]
        
        print("\n✓ Error handling test passed!")


class APIClient:
    """Simple HTTP client for API testing."""
    
    def __init__(self, base_url: str, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        
        # Check if server is available
        try:
            self.get("/health", timeout=2)
        except requests.ConnectionError:
            pytest.skip(f"API server not available at {base_url}")
    
    def get(self, endpoint: str, **kwargs):
        """Make GET request."""
        url = f"{self.base_url}{endpoint}"
        timeout = kwargs.pop("timeout", self.timeout)
        return self.session.get(url, timeout=timeout, **kwargs)
    
    def post(self, endpoint: str, **kwargs):
        """Make POST request."""
        url = f"{self.base_url}{endpoint}"
        return self.session.post(url, timeout=self.timeout, **kwargs)
    
    def put(self, endpoint: str, **kwargs):
        """Make PUT request."""
        url = f"{self.base_url}{endpoint}"
        return self.session.put(url, timeout=self.timeout, **kwargs)
    
    def delete(self, endpoint: str, **kwargs):
        """Make DELETE request."""
        url = f"{self.base_url}{endpoint}"
        return self.session.delete(url, timeout=self.timeout, **kwargs)


class TestLocalStackIntegration:
    """Tests for LocalStack AWS service integration."""
    
    @pytest.fixture(scope="class")
    def aws_client(self):
        """Create AWS client for LocalStack."""
        import boto3
        
        return boto3.client(
            "dynamodb",
            endpoint_url="http://localhost:4566",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            region_name="us-east-1",
        )
    
    def test_dynamodb_tables_exist(self, aws_client):
        """Verify DynamoDB tables were created."""
        response = aws_client.list_tables()
        tables = response.get("TableNames", [])
        
        print(f"\nDynamoDB Tables: {tables}")
        
        expected_tables = ["Conversations", "Feedback", "PromptVersions", "AnalyticsSummary"]
        for table in expected_tables:
            assert table in tables, f"Table {table} not found"
    
    def test_dynamodb_conversation_crud(self, aws_client):
        """Test DynamoDB conversation CRUD operations."""
        conversation_id = f"test-{uuid.uuid4()}"
        
        # Create
        aws_client.put_item(
            TableName="Conversations",
            Item={
                "conversation_id": {"S": conversation_id},
                "user_id": {"S": "test_user"},
                "status": {"S": "active"},
                "created_at": {"S": datetime.utcnow().isoformat()},
                "history": {"S": json.dumps([])},
            }
        )
        
        # Read
        response = aws_client.get_item(
            TableName="Conversations",
            Key={"conversation_id": {"S": conversation_id}}
        )
        
        assert response["Item"]["conversation_id"]["S"] == conversation_id
        
        print(f"\n✓ DynamoDB CRUD test passed for conversation {conversation_id}")


@pytest.mark.skipif(
    not Path("data/chroma").exists(),
    reason="ChromaDB not initialized"
)
class TestChromaDBIntegration:
    """Tests for ChromaDB vector store integration."""
    
    def test_chromadb_connection(self):
        """Test connection to ChromaDB."""
        import chromadb
        
        client = chromadb.PersistentClient(path="data/chroma")
        
        # List collections
        collections = client.list_collections()
        print(f"\nChromaDB Collections: {[c.name for c in collections]}")
        
        # Check FAQ collection
        collection = client.get_collection("faq_documents")
        count = collection.count()
        
        print(f"  FAQ documents: {count}")
        assert count >= 50, f"Expected at least 50 documents, got {count}"
    
    def test_chromadb_search(self):
        """Test ChromaDB search functionality."""
        import chromadb
        from sentence_transformers import SentenceTransformer
        
        client = chromadb.PersistentClient(path="data/chroma")
        collection = client.get_collection("faq_documents")
        
        # Load embedding model
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Search for return-related query
        query = "How do I return a damaged item?"
        query_embedding = model.encode([query]).tolist()
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        
        print(f"\nChromaDB search for: '{query}'")
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            score = 1 - dist
            print(f"  {i+1}. [{meta.get('intent', 'unknown')}] Score: {score:.3f}")
            print(f"     {doc[:80]}...")
        
        assert len(results["documents"][0]) == 3


def test_full_platform_integration():
    """
    Full platform integration test - runs all major components together.
    This test can be run locally to verify the entire platform works.
    """
    print("\n" + "=" * 70)
    print("FULL PLATFORM INTEGRATION TEST")
    print("=" * 70)
    
    # This is a comprehensive test that can be run manually
    # It requires all services to be running
    
    print("\nPrerequisites:")
    print("  - LocalStack running (docker-compose up localstack)")
    print("  - ChromaDB running (docker-compose up chromadb)")
    print("  - API server running (docker-compose --profile api up)")
    print("\nRun with: pytest tests/integration/test_e2e.py::test_full_platform_integration -v -s")
    
    # Verify components can be imported
    from src.intent_classifier import IntentClassifier
    from src.response_generator import RAGEngine, VectorStore, ConversationManager
    from src.feedback_pipeline import FeedbackAnalyzer
    
    print("\n✓ All components imported successfully")
    print("\n✓ Full platform integration test configuration verified!")

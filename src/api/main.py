"""FastAPI application for local development and testing."""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances
_services = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info("Starting up Customer Service AI API...")
    
    # Initialize services
    from src.intent_classifier import IntentClassifier
    from src.response_generator import RAGEngine, ConversationManager, VectorStore
    from src.feedback_pipeline import FeedbackAnalyzer
    
    # Load intent classifier
    model_path = os.environ.get("INTENT_MODEL_PATH", "data/models/onnx")
    if os.path.exists(model_path):
        _services["intent_classifier"] = IntentClassifier(model_path=model_path, use_onnx=True)
        logger.info("Intent classifier loaded")
    
    # Initialize RAG engine
    _services["vector_store"] = VectorStore()
    _services["conversation_manager"] = ConversationManager(use_dynamodb=False)
    _services["rag_engine"] = RAGEngine(
        vector_store=_services["vector_store"],
        conversation_manager=_services["conversation_manager"],
    )
    logger.info("RAG engine initialized")
    
    # Initialize feedback analyzer
    _services["feedback_analyzer"] = FeedbackAnalyzer()
    logger.info("Feedback analyzer initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Customer Service AI API",
    description="Production-ready MVP for Customer Service AI Agent Platform",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models

class ChatRequest(BaseModel):
    message: str = Field(..., description="Customer message")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")
    user_id: Optional[str] = Field(None, description="User identifier")
    intent: Optional[str] = Field(None, description="Pre-classified intent (optional)")


class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    intent: str
    confidence: float
    model_used: str
    turn_number: int
    retrieved_context: Optional[List[Dict]] = None


class IntentRequest(BaseModel):
    text: str


class IntentResponse(BaseModel):
    intent: str
    confidence: float
    all_scores: Dict[str, float]


class FeedbackRequest(BaseModel):
    conversation_id: str
    text: str
    rating: str = Field(..., description="positive, negative, or neutral")
    turn_index: Optional[int] = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    sentiment: str
    confidence: float
    keywords: List[str]


class AnalyticsQuery(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    intent: Optional[str] = None


# Middleware for request timing
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "intent_classifier": "intent_classifier" in _services,
            "rag_engine": "rag_engine" in _services,
            "feedback_analyzer": "feedback_analyzer" in _services,
        },
    }


# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and generate a response."""
    try:
        # Classify intent if not provided
        intent = request.intent
        confidence = 1.0
        
        if not intent and "intent_classifier" in _services:
            intent_result = _services["intent_classifier"].predict(request.message)[0]
            intent = intent_result["intent"]
            confidence = intent_result["confidence"]
        elif not intent:
            intent = "general_inquiry"
        
        # Generate response
        rag_engine = _services["rag_engine"]
        result = rag_engine.generate_response(
            query=request.message,
            conversation_id=request.conversation_id,
            intent=intent,
            user_id=request.user_id,
        )
        
        return ChatResponse(
            conversation_id=result["conversation_id"],
            response=result["response"],
            intent=intent,
            confidence=confidence,
            model_used=result.get("model_used", "unknown"),
            turn_number=result.get("turn_number", 1),
            retrieved_context=result.get("retrieved_context", []),
        )
        
    except Exception as e:
        logger.exception("Error in chat endpoint")
        raise HTTPException(status_code=500, detail=str(e))


# Intent classification endpoint
@app.post("/classify", response_model=IntentResponse)
async def classify_intent(request: IntentRequest):
    """Classify the intent of a message."""
    if "intent_classifier" not in _services:
        raise HTTPException(status_code=503, detail="Intent classifier not available")
    
    try:
        result = _services["intent_classifier"].predict(request.text)[0]
        return IntentResponse(**result)
    except Exception as e:
        logger.exception("Error in classify endpoint")
        raise HTTPException(status_code=500, detail=str(e))


# Feedback endpoint
@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for analysis."""
    try:
        analyzer = _services["feedback_analyzer"]
        analysis = analyzer.analyze_feedback(request.text)
        
        # Store feedback in conversation if applicable
        if request.turn_index and "conversation_manager" in _services:
            rag_engine = _services["rag_engine"]
            rag_engine.add_feedback(
                conversation_id=request.conversation_id,
                turn_index=request.turn_index,
                feedback="positive" if request.rating == "positive" else "negative",
            )
        
        import uuid
        return FeedbackResponse(
            feedback_id=str(uuid.uuid4()),
            sentiment=analysis["sentiment"]["label"],
            confidence=analysis["sentiment"]["confidence"],
            keywords=[kw["phrase"] for kw in analysis["keywords"][:5]],
        )
        
    except Exception as e:
        logger.exception("Error in feedback endpoint")
        raise HTTPException(status_code=500, detail=str(e))


# Analytics endpoint
@app.get("/analytics")
async def get_analytics():
    """Get analytics summary."""
    try:
        # Get conversation statistics
        conv_manager = _services.get("conversation_manager")
        total_conversations = len(conv_manager._local_store) if conv_manager else 0
        
        return {
            "total_conversations": total_conversations,
            "active_conversations": total_conversations,  # Simplified
            "average_response_time_ms": 0,  # Would be tracked in production
        }
    except Exception as e:
        logger.exception("Error in analytics endpoint")
        raise HTTPException(status_code=500, detail=str(e))


# Conversation history endpoint
@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history."""
    try:
        conv_manager = _services.get("conversation_manager")
        if not conv_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
        
        conversation = conv_manager.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return conversation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting conversation")
        raise HTTPException(status_code=500, detail=str(e))


def start():
    """Start the API server."""
    import uvicorn
    port = int(os.environ.get("API_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    start()

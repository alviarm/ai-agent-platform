"""Vector Store implementation using ChromaDB and FAISS."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .config import (
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    TOP_K_RETRIEVAL,
    VECTOR_STORE_PATH,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store for RAG retrieval using ChromaDB."""
    
    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        embedding_model: str = EMBEDDING_MODEL,
        collection_name: str = "faq_documents",
    ):
        """Initialize vector store.
        
        Args:
            persist_dir: Directory to persist ChromaDB data
            embedding_model: Sentence transformer model for embeddings
            collection_name: Name of the ChromaDB collection
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize ChromaDB (new API for version 0.4+)
        try:
            # Try new API first (ChromaDB 0.4+)
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        except (AttributeError, TypeError):
            # Fallback to old API (ChromaDB < 0.4)
            self.client = chromadb.Client(
                Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=str(self.persist_dir),
                )
            )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        
        logger.info(f"VectorStore initialized with {self.collection.count()} documents")
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of text documents
            metadatas: Optional metadata for each document
            ids: Optional IDs for documents (generated if not provided)
            
        Returns:
            List of document IDs
        """
        if ids is None:
            ids = [self._generate_id(doc) for doc in documents]
        
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        
        # Persist (only needed for old ChromaDB API)
        if hasattr(self.client, 'persist'):
            self.client.persist()
        
        logger.info(f"Added {len(documents)} documents to vector store")
        return ids
    
    def search(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        filter_dict: Optional[Dict] = None,
    ) -> List[Dict]:
        """Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Optional filter for metadata
            
        Returns:
            List of result dictionaries with document, metadata, and score
        """
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=filter_dict,
            include=["documents", "metadatas", "distances"],
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],  # Convert distance to similarity
            })
        
        return formatted_results
    
    def delete_document(self, doc_id: str):
        """Delete a document by ID."""
        self.collection.delete(ids=[doc_id])
        self.client.persist()
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents in the store."""
        results = self.collection.get()
        
        documents = []
        for i in range(len(results["ids"])):
            documents.append({
                "id": results["ids"][i],
                "document": results["documents"][i],
                "metadata": results["metadatas"][i],
            })
        return documents
    
    def clear(self):
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
    
    def _generate_id(self, text: str) -> str:
        """Generate unique ID for document."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def chunk_document(
        self,
        text: str,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> List[str]:
        """Split document into chunks.
        
        Args:
            text: Document text
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Simple word-based chunking
        words = text.split()
        chunks = []
        
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = end - chunk_overlap
        
        return chunks
    
    def load_faq_documents(self, faq_path: str):
        """Load FAQ documents from JSON file.
        
        Expected format:
        [
            {
                "question": "What is your return policy?",
                "answer": "We accept returns within 30 days...",
                "category": "returns",
                "intent": "return"
            },
            ...
        ]
        """
        with open(faq_path, "r") as f:
            faqs = json.load(f)
        
        documents = []
        metadatas = []
        
        for faq in faqs:
            # Combine question and answer for embedding
            content = f"Q: {faq['question']}\nA: {faq['answer']}"
            documents.append(content)
            metadatas.append({
                "question": faq["question"],
                "answer": faq["answer"],
                "category": faq.get("category", "general"),
                "intent": faq.get("intent", "general_inquiry"),
            })
        
        self.add_documents(documents, metadatas)
        logger.info(f"Loaded {len(faqs)} FAQ documents")

"""
Vector Store Implementation using ChromaDB
Handles document embeddings, storage, and similarity search
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import uuid


class VectorStore:
    """
    Vector store for managing document embeddings and retrieval
    Uses ChromaDB for persistent storage and sentence-transformers for embeddings
    """
    
    def __init__(self, persist_directory: str = "./chroma_db", 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 collection_name: str = "financial_documents"):
        """
        Initialize the vector store
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            embedding_model: Sentence transformer model name
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"✅ Vector store initialized with {self.collection.count()} documents")
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """
        Add documents to the vector store
        
        Args:
            texts: List of text chunks to add
            metadatas: Optional list of metadata dicts for each chunk
            
        Returns:
            List of document IDs
        """
        if not texts:
            return []
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False).tolist()
        
        # Generate IDs
        ids = [str(uuid.uuid4()) for _ in texts]
        
        # Prepare metadatas
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of dicts with 'text', 'metadata', 'score', 'id'
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False).tolist()[0]
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'text': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'id': results['ids'][0][i] if results['ids'] else None
                })
        
        return formatted_results
    
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents by IDs"""
        self.collection.delete(ids=ids)
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        return {
            'total_documents': self.collection.count(),
            'embedding_model': self.embedding_model_name,
            'collection_name': self.collection.name
        }
    
    def reset(self) -> None:
        """Clear all documents from the collection"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def close(self) -> None:
        """Cleanup resources"""
        # ChromaDB client doesn't need explicit closing
        pass


class HybridSearch:
    """
    Hybrid search combining dense (semantic) and sparse (keyword) retrieval
    For production use - demonstrates advanced RAG technique
    """
    
    def __init__(self, vector_store: VectorStore, bm25_weight: float = 0.3):
        """
        Initialize hybrid search
        
        Args:
            vector_store: VectorStore instance
            bm25_weight: Weight for BM25 scores (0-1), rest goes to semantic
        """
        self.vector_store = vector_store
        self.bm25_weight = bm25_weight
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Perform hybrid search
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            Reranked results combining semantic and keyword search
        """
        # For now, just use semantic search
        # In production, combine with BM25 using Reciprocal Rank Fusion
        return self.vector_store.search(query, top_k)

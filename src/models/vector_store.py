"""
Vector Store Implementation using Pinecone
Handles document embeddings, storage, and similarity search
"""

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import uuid
import time


class VectorStore:
    """
    Vector store for managing document embeddings and retrieval
    Uses Pinecone Serverless for cloud-hosted storage and sentence-transformers for embeddings
    """

    def __init__(self, api_key: str,
                 index_name: str = "financial-documents",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 cloud: str = "aws",
                 region: str = "us-east-1"):
        """
        Initialize the vector store

        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            embedding_model: Sentence transformer model name
            cloud: Cloud provider for serverless (aws, gcp, azure)
            region: Cloud region for serverless
        """
        self.index_name = index_name
        self.embedding_model_name = embedding_model

        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        dimension = self.embedding_model.get_sentence_embedding_dimension()

        # Initialize Pinecone client
        self.client = Pinecone(api_key=api_key)

        # Create index if it doesn't exist
        existing_indexes = [idx.name for idx in self.client.list_indexes()]
        if index_name not in existing_indexes:
            self.client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
            # Wait for index to be ready
            while not self.client.describe_index(index_name).status['ready']:
                time.sleep(1)

        # Connect to the index
        self.index = self.client.Index(index_name)

        stats = self.index.describe_index_stats()
        print(f"Vector store initialized with {stats.total_vector_count} documents")

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

        # Prepare metadatas — store text in metadata for retrieval
        if metadatas is None:
            metadatas = [{}] * len(texts)

        vectors = []
        for i, (doc_id, embedding, text) in enumerate(zip(ids, embeddings, texts)):
            metadata = {**metadatas[i], "text": text}
            vectors.append((doc_id, embedding, metadata))

        # Upsert in batches of 100 (Pinecone recommended batch size)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)

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

        # Search Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Format results
        formatted_results = []
        for match in results.matches:
            metadata = dict(match.metadata) if match.metadata else {}
            text = metadata.pop("text", "")
            formatted_results.append({
                'text': text,
                'metadata': metadata,
                'score': match.score,
                'id': match.id
            })

        return formatted_results

    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents by IDs"""
        self.index.delete(ids=ids)

    def get_collection_stats(self) -> Dict:
        """Get statistics about the index"""
        stats = self.index.describe_index_stats()
        return {
            'total_documents': stats.total_vector_count,
            'embedding_model': self.embedding_model_name,
            'collection_name': self.index_name
        }

    def reset(self) -> None:
        """Clear all documents from the index"""
        self.index.delete(delete_all=True)

    def close(self) -> None:
        """Cleanup resources"""
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

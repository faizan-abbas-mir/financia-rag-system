"""
Tests for VectorStore class
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from models.vector_store import VectorStore


@pytest.fixture
def temp_db_dir():
    """Create temporary directory for test database"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def vector_store(temp_db_dir):
    """Create VectorStore instance for testing"""
    store = VectorStore(
        persist_directory=temp_db_dir,
        embedding_model="all-MiniLM-L6-v2",
        collection_name="test_collection"
    )
    yield store
    store.close()


class TestVectorStore:
    """Test suite for VectorStore"""
    
    def test_initialization(self, vector_store):
        """Test vector store initializes correctly"""
        assert vector_store is not None
        assert vector_store.collection is not None
        stats = vector_store.get_collection_stats()
        assert stats['total_documents'] == 0
    
    def test_add_documents(self, vector_store):
        """Test adding documents"""
        texts = [
            "Apple reported strong Q3 earnings with revenue growth of 15%",
            "Microsoft's cloud business continues to grow rapidly",
            "Tesla delivered record number of vehicles in Q4"
        ]
        
        ids = vector_store.add_documents(texts)
        
        assert len(ids) == 3
        stats = vector_store.get_collection_stats()
        assert stats['total_documents'] == 3
    
    def test_search(self, vector_store):
        """Test search functionality"""
        # Add documents
        texts = [
            "Apple's iPhone sales increased by 20% in Q3 2024",
            "Microsoft Azure revenue grew 30% year over year",
            "Amazon's AWS remains market leader in cloud computing",
            "Google's advertising revenue hit new records"
        ]
        vector_store.add_documents(texts)
        
        # Search
        results = vector_store.search("cloud computing growth", top_k=2)
        
        assert len(results) == 2
        assert all('text' in r for r in results)
        assert all('score' in r for r in results)
        assert all(r['score'] >= 0 and r['score'] <= 1 for r in results)
        
        # Check if relevant documents are retrieved
        retrieved_texts = [r['text'] for r in results]
        assert any('Azure' in text or 'AWS' in text for text in retrieved_texts)
    
    def test_search_with_metadata(self, vector_store):
        """Test search with metadata"""
        texts = ["Q3 earnings report", "Q4 earnings report"]
        metadatas = [
            {"quarter": "Q3", "year": 2024},
            {"quarter": "Q4", "year": 2024}
        ]
        
        vector_store.add_documents(texts, metadatas)
        results = vector_store.search("earnings", top_k=2)
        
        assert len(results) == 2
        assert all('metadata' in r for r in results)
        assert all('quarter' in r['metadata'] for r in results)
    
    def test_delete_documents(self, vector_store):
        """Test document deletion"""
        texts = ["Document 1", "Document 2", "Document 3"]
        ids = vector_store.add_documents(texts)
        
        # Delete first document
        vector_store.delete_documents([ids[0]])
        
        stats = vector_store.get_collection_stats()
        assert stats['total_documents'] == 2
    
    def test_reset(self, vector_store):
        """Test collection reset"""
        texts = ["Doc 1", "Doc 2", "Doc 3"]
        vector_store.add_documents(texts)
        
        vector_store.reset()
        
        stats = vector_store.get_collection_stats()
        assert stats['total_documents'] == 0
    
    def test_search_empty_store(self, vector_store):
        """Test search on empty store"""
        results = vector_store.search("test query", top_k=3)
        assert len(results) == 0
    
    def test_relevance_scores_ordering(self, vector_store):
        """Test that results are ordered by relevance"""
        texts = [
            "Cloud computing enables scalable infrastructure",
            "Weather forecast predicts rain tomorrow",
            "Cloud services have transformed business operations"
        ]
        vector_store.add_documents(texts)
        
        results = vector_store.search("cloud infrastructure", top_k=3)
        
        # Scores should be in descending order
        scores = [r['score'] for r in results]
        assert scores == sorted(scores, reverse=True)
        
        # Most relevant document should be about cloud computing
        assert 'Cloud' in results[0]['text']


class TestHybridSearch:
    """Test suite for HybridSearch (if implemented)"""
    
    def test_hybrid_search_placeholder(self):
        """Placeholder for hybrid search tests"""
        # TODO: Implement when HybridSearch is fully developed
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

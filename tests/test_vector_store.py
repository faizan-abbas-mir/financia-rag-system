"""
Tests for VectorStore class (Pinecone)
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from models.vector_store import VectorStore


# --- Helpers to build mock Pinecone responses ---

def _make_match(id, score, metadata):
    """Create a mock Pinecone match object."""
    match = MagicMock()
    match.id = id
    match.score = score
    match.metadata = metadata
    return match


def _make_query_response(matches):
    resp = MagicMock()
    resp.matches = matches
    return resp


def _make_stats(total_vector_count=0):
    stats = MagicMock()
    stats.total_vector_count = total_vector_count
    return stats


def _make_index_description(ready=True):
    desc = MagicMock()
    desc.status = {'ready': ready}
    return desc


def _make_index_info(name):
    info = MagicMock()
    info.name = name
    return info


# --- Fixtures ---

@pytest.fixture
def mock_pinecone():
    """Patch Pinecone client and SentenceTransformer."""
    with patch("models.vector_store.Pinecone") as MockPinecone, \
         patch("models.vector_store.SentenceTransformer") as MockST:

        # Mock SentenceTransformer
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.1] * 384])
        MockST.return_value = mock_model

        # Mock Pinecone client
        mock_client = MagicMock()
        mock_index = MagicMock()

        mock_client.list_indexes.return_value = [_make_index_info("financial-documents")]
        mock_client.describe_index.return_value = _make_index_description(ready=True)
        mock_client.Index.return_value = mock_index
        mock_index.describe_index_stats.return_value = _make_stats(0)
        mock_index.query.return_value = _make_query_response([])

        MockPinecone.return_value = mock_client

        yield {
            "client": mock_client,
            "index": mock_index,
            "model": mock_model,
            "PineconeClass": MockPinecone,
            "STClass": MockST,
        }


@pytest.fixture
def vector_store(mock_pinecone):
    """Create VectorStore instance for testing."""
    store = VectorStore(
        api_key="test-api-key",
        index_name="financial-documents",
        embedding_model="all-MiniLM-L6-v2"
    )
    yield store
    store.close()


class TestVectorStore:
    """Test suite for VectorStore"""

    def test_initialization(self, vector_store, mock_pinecone):
        """Test vector store initializes correctly"""
        assert vector_store is not None
        assert vector_store.index is not None
        mock_pinecone["PineconeClass"].assert_called_once_with(api_key="test-api-key")
        mock_pinecone["client"].Index.assert_called_once_with("financial-documents")

    def test_initialization_creates_index_if_missing(self, mock_pinecone):
        """Test that a new index is created when it doesn't exist"""
        mock_pinecone["client"].list_indexes.return_value = []  # no existing indexes

        store = VectorStore(
            api_key="test-api-key",
            index_name="new-index",
            embedding_model="all-MiniLM-L6-v2"
        )

        mock_pinecone["client"].create_index.assert_called_once()
        call_kwargs = mock_pinecone["client"].create_index.call_args
        assert call_kwargs.kwargs["name"] == "new-index"
        assert call_kwargs.kwargs["dimension"] == 384
        assert call_kwargs.kwargs["metric"] == "cosine"

    def test_add_documents(self, vector_store, mock_pinecone):
        """Test adding documents"""
        texts = [
            "Apple reported strong Q3 earnings with revenue growth of 15%",
            "Microsoft's cloud business continues to grow rapidly",
            "Tesla delivered record number of vehicles in Q4"
        ]

        # Mock encode to return one embedding per text
        mock_pinecone["model"].encode.return_value = MagicMock(
            tolist=lambda: [[0.1] * 384] * 3
        )

        ids = vector_store.add_documents(texts)

        assert len(ids) == 3
        mock_pinecone["index"].upsert.assert_called_once()
        upsert_call = mock_pinecone["index"].upsert.call_args
        vectors = upsert_call.kwargs["vectors"]
        assert len(vectors) == 3
        # Verify text is stored in metadata
        assert vectors[0][2]["text"] == texts[0]

    def test_add_documents_empty(self, vector_store, mock_pinecone):
        """Test adding empty list returns empty"""
        ids = vector_store.add_documents([])
        assert ids == []
        mock_pinecone["index"].upsert.assert_not_called()

    def test_search(self, vector_store, mock_pinecone):
        """Test search functionality"""
        mock_pinecone["model"].encode.return_value = MagicMock(
            tolist=lambda: [[0.1] * 384]
        )
        mock_pinecone["index"].query.return_value = _make_query_response([
            _make_match("id1", 0.92, {"text": "Microsoft Azure revenue grew 30%", "filename": "report.pdf"}),
            _make_match("id2", 0.85, {"text": "Amazon's AWS remains market leader", "filename": "report.pdf"}),
        ])

        results = vector_store.search("cloud computing growth", top_k=2)

        assert len(results) == 2
        assert all('text' in r for r in results)
        assert all('score' in r for r in results)
        assert results[0]['score'] == 0.92
        assert results[1]['score'] == 0.85
        assert any('Azure' in r['text'] or 'AWS' in r['text'] for r in results)

    def test_search_with_metadata(self, vector_store, mock_pinecone):
        """Test search with metadata"""
        mock_pinecone["model"].encode.return_value = MagicMock(
            tolist=lambda: [[0.1] * 384]
        )
        mock_pinecone["index"].query.return_value = _make_query_response([
            _make_match("id1", 0.9, {"text": "Q3 earnings report", "quarter": "Q3", "year": 2024}),
            _make_match("id2", 0.8, {"text": "Q4 earnings report", "quarter": "Q4", "year": 2024}),
        ])

        results = vector_store.search("earnings", top_k=2)

        assert len(results) == 2
        assert all('metadata' in r for r in results)
        assert all('quarter' in r['metadata'] for r in results)
        # text should be extracted from metadata, not remain in it
        assert 'text' not in results[0]['metadata']

    def test_delete_documents(self, vector_store, mock_pinecone):
        """Test document deletion"""
        vector_store.delete_documents(["id1", "id2"])
        mock_pinecone["index"].delete.assert_called_once_with(ids=["id1", "id2"])

    def test_reset(self, vector_store, mock_pinecone):
        """Test collection reset"""
        vector_store.reset()
        mock_pinecone["index"].delete.assert_called_once_with(delete_all=True)

    def test_search_empty_store(self, vector_store, mock_pinecone):
        """Test search on empty store"""
        mock_pinecone["model"].encode.return_value = MagicMock(
            tolist=lambda: [[0.1] * 384]
        )
        mock_pinecone["index"].query.return_value = _make_query_response([])

        results = vector_store.search("test query", top_k=3)
        assert len(results) == 0

    def test_relevance_scores_ordering(self, vector_store, mock_pinecone):
        """Test that results are ordered by relevance"""
        mock_pinecone["model"].encode.return_value = MagicMock(
            tolist=lambda: [[0.1] * 384]
        )
        # Pinecone returns results already sorted by score descending
        mock_pinecone["index"].query.return_value = _make_query_response([
            _make_match("id1", 0.95, {"text": "Cloud computing enables scalable infrastructure"}),
            _make_match("id2", 0.88, {"text": "Cloud services have transformed business operations"}),
            _make_match("id3", 0.42, {"text": "Weather forecast predicts rain tomorrow"}),
        ])

        results = vector_store.search("cloud infrastructure", top_k=3)

        scores = [r['score'] for r in results]
        assert scores == sorted(scores, reverse=True)
        assert 'Cloud' in results[0]['text']

    def test_get_collection_stats(self, vector_store, mock_pinecone):
        """Test getting collection statistics"""
        mock_pinecone["index"].describe_index_stats.return_value = _make_stats(42)

        stats = vector_store.get_collection_stats()

        assert stats['total_documents'] == 42
        assert stats['embedding_model'] == 'all-MiniLM-L6-v2'
        assert stats['collection_name'] == 'financial-documents'


class TestHybridSearch:
    """Test suite for HybridSearch (if implemented)"""

    def test_hybrid_search_placeholder(self):
        """Placeholder for hybrid search tests"""
        # TODO: Implement when HybridSearch is fully developed
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

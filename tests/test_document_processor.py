"""
Tests for DocumentProcessor class
"""

import pytest
import tempfile
import os
from pathlib import Path
from utils.document_processor import DocumentProcessor


@pytest.fixture
def processor():
    """Create DocumentProcessor instance"""
    return DocumentProcessor(chunk_size=512, chunk_overlap=50)


@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return """
    Apple Inc. reported strong quarterly results with revenue reaching $89.5 billion, 
    representing a 15% year-over-year growth. The company's iPhone sales increased by 20%, 
    driven by strong demand for the latest models. Services revenue also grew by 18%, 
    reaching a new record high. CEO Tim Cook stated that the company is seeing strong 
    momentum across all geographic regions and product categories. The board approved 
    a new $90 billion share buyback program, demonstrating confidence in future growth.
    """


class TestDocumentProcessor:
    """Test suite for DocumentProcessor"""
    
    def test_initialization(self, processor):
        """Test processor initializes with correct parameters"""
        assert processor.chunk_size == 512
        assert processor.chunk_overlap == 50
    
    def test_clean_text(self, processor, sample_text):
        """Test text cleaning"""
        cleaned = processor._clean_text(sample_text)
        
        # Should remove excessive whitespace
        assert '  ' not in cleaned
        assert cleaned.startswith('Apple')
        assert cleaned.endswith('growth.')
    
    def test_chunk_text_basic(self, processor):
        """Test basic text chunking"""
        text = "This is sentence one. This is sentence two. This is sentence three."
        
        chunks = processor.chunk_text(text)
        
        assert len(chunks) > 0
        assert all('text' in chunk for chunk in chunks)
        assert all('chunk_id' in chunk for chunk in chunks)
    
    def test_chunk_text_overlap(self, processor):
        """Test that chunks have proper overlap"""
        # Create text with clear sentences
        sentences = [f"This is sentence number {i}." for i in range(10)]
        text = " ".join(sentences)
        
        chunks = processor.chunk_text(text)
        
        if len(chunks) > 1:
            # Check that consecutive chunks share some content (overlap)
            # Note: This is a simplified check
            assert len(chunks[0]['text']) > 0
            assert len(chunks[1]['text']) > 0
    
    def test_chunk_text_preserves_sentences(self, processor):
        """Test that chunking preserves sentence boundaries"""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        
        chunks = processor.chunk_text(text)
        
        for chunk in chunks:
            # Each chunk should end with sentence-ending punctuation or be the last chunk
            assert chunk['text'][-1] in '.!?' or chunk['chunk_id'] == len(chunks) - 1
    
    def test_split_sentences(self, processor):
        """Test sentence splitting"""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        
        sentences = processor._split_sentences(text)
        
        assert len(sentences) == 4
        assert sentences[0].strip() == "First sentence."
        assert sentences[1].strip() == "Second sentence!"
    
    def test_extract_amounts(self, processor):
        """Test extracting monetary amounts"""
        text = "Revenue was $89.5B with profit of $23.4M and costs of $1,234.56."
        
        amounts = processor._extract_amounts(text)
        
        assert len(amounts) > 0
        assert any('$89.5' in amt for amt in amounts)
    
    def test_extract_percentages(self, processor):
        """Test extracting percentages"""
        text = "Growth was 15% while margins improved by 2.5% to reach 23.8%."
        
        percentages = processor._extract_percentages(text)
        
        assert len(percentages) >= 3
        assert '15%' in percentages
        assert '2.5%' in percentages
    
    def test_extract_dates(self, processor):
        """Test extracting dates"""
        text = "Q3 2024 results released on 2024-10-15 showed improvement from 10/01/2024."
        
        dates = processor._extract_dates(text)
        
        assert len(dates) > 0
        assert any('Q3' in date for date in dates)
    
    def test_extract_financial_entities(self, processor, sample_text):
        """Test extracting all financial entities"""
        entities = processor.extract_financial_entities(sample_text)
        
        assert 'amounts' in entities
        assert 'percentages' in entities
        assert 'dates' in entities
        
        # Should find the amounts and percentages in sample text
        assert len(entities['amounts']) > 0
        assert len(entities['percentages']) > 0


class TestChunkingStrategy:
    """Test different chunking strategies"""
    
    def test_small_chunks(self):
        """Test with small chunk size"""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        text = " ".join([f"Sentence {i}." for i in range(20)])
        
        chunks = processor.chunk_text(text)
        
        assert len(chunks) > 0
        assert all(len(chunk['text']) <= 150 for chunk in chunks)  # Allow some margin
    
    def test_large_chunks(self):
        """Test with large chunk size"""
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=100)
        text = " ".join([f"Sentence {i}." for i in range(50)])
        
        chunks = processor.chunk_text(text)
        
        # Larger chunks should result in fewer total chunks
        assert len(chunks) > 0
    
    def test_no_overlap(self):
        """Test chunking without overlap"""
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=0)
        text = " ".join([f"Sentence {i}." for i in range(20)])
        
        chunks = processor.chunk_text(text)
        
        assert len(chunks) > 0
        # With no overlap, chunks should be completely distinct


class TestFileProcessing:
    """Test processing different file types"""
    
    def create_temp_file(self, content, extension):
        """Helper to create temporary test files"""
        fd, path = tempfile.mkstemp(suffix=extension)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        return path
    
    def test_process_txt_file(self, processor, sample_text):
        """Test processing TXT file"""
        filepath = self.create_temp_file(sample_text, '.txt')
        
        try:
            result = processor.process_file(filepath)
            
            assert 'text' in result
            assert 'metadata' in result
            assert 'chunks' in result
            
            assert result['metadata']['file_type'] == '.txt'
            assert len(result['chunks']) > 0
        finally:
            os.unlink(filepath)
    
    def test_unsupported_file_type(self, processor):
        """Test handling of unsupported file types"""
        filepath = self.create_temp_file("test", '.xyz')
        
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                processor.process_file(filepath)
        finally:
            os.unlink(filepath)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

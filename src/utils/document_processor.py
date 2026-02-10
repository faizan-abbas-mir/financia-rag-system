"""
Document Processing Utilities
Handles document parsing, chunking, and preprocessing
"""

import os
import re
from typing import List, Dict, Optional
from pathlib import Path

# Import document parsers
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None


class DocumentProcessor:
    """
    Processes documents for RAG system
    Supports PDF, DOCX, TXT formats
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize document processor
        
        Args:
            chunk_size: Target size for text chunks (in characters)
            chunk_overlap: Overlap between chunks (in characters)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_file(self, file_path: str) -> Dict:
        """
        Process a document file and extract text
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dict with 'text', 'metadata', 'chunks'
        """
        file_extension = Path(file_path).suffix.lower()
        
        # Extract text based on file type
        if file_extension == '.pdf':
            text = self._extract_pdf(file_path)
        elif file_extension == '.docx':
            text = self._extract_docx(file_path)
        elif file_extension == '.txt':
            text = self._extract_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Clean text
        text = self._clean_text(text)
        
        # Create chunks
        chunks = self.chunk_text(text)
        
        # Create metadata
        metadata = {
            'filename': Path(file_path).name,
            'file_type': file_extension,
            'file_size': os.path.getsize(file_path),
            'num_chunks': len(chunks)
        }
        
        return {
            'text': text,
            'metadata': metadata,
            'chunks': chunks
        }
    
    def _extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""
        if PdfReader is None:
            raise ImportError("PyPDF2 not installed. Run: pip install PyPDF2")
        
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def _extract_docx(self, file_path: str) -> str:
        """Extract text from DOCX"""
        if DocxDocument is None:
            raise ImportError("python-docx not installed. Run: pip install python-docx")
        
        doc = DocxDocument(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    
    def _extract_txt(self, file_path: str) -> str:
        """Extract text from TXT"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might cause issues
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        return text.strip()
    
    def chunk_text(self, text: str) -> List[Dict]:
        """
        Split text into overlapping chunks
        Uses sentence-aware splitting for better semantic coherence
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of chunk dicts with 'text', 'chunk_id', 'start_idx'
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        start_idx = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk size and we have content
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'chunk_id': len(chunks),
                    'start_idx': start_idx,
                    'end_idx': start_idx + len(chunk_text)
                })
                
                # Calculate overlap for next chunk
                overlap_text = ' '.join(current_chunk[-2:]) if len(current_chunk) >= 2 else ''
                current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                current_length = len(overlap_text) + sentence_length
                start_idx += len(chunk_text) - len(overlap_text)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'chunk_id': len(chunks),
                'start_idx': start_idx,
                'end_idx': start_idx + len(chunk_text)
            })
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        Uses simple regex - for production, use nltk or spacy
        """
        # Basic sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def extract_financial_entities(self, text: str) -> Dict:
        """
        Extract financial entities from text
        Simple regex-based extraction - for production use NER models
        
        Returns:
            Dict with extracted entities: amounts, percentages, dates, companies
        """
        entities = {
            'amounts': self._extract_amounts(text),
            'percentages': self._extract_percentages(text),
            'dates': self._extract_dates(text),
        }
        
        return entities
    
    def _extract_amounts(self, text: str) -> List[str]:
        """Extract monetary amounts"""
        pattern = r'\$[\d,]+\.?\d*[BMK]?'
        return re.findall(pattern, text)
    
    def _extract_percentages(self, text: str) -> List[str]:
        """Extract percentages"""
        pattern = r'\d+\.?\d*%'
        return re.findall(pattern, text)
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates (simple patterns)"""
        patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY
            r'\d{4}-\d{2}-\d{2}',         # YYYY-MM-DD
            r'Q[1-4]\s*\d{4}',            # Q1 2024
        ]
        dates = []
        for pattern in patterns:
            dates.extend(re.findall(pattern, text))
        return dates


class ChunkingStrategy:
    """
    Advanced chunking strategies for different document types
    Demonstrates production-ready RAG techniques
    """
    
    @staticmethod
    def semantic_chunking(text: str, embedding_model, similarity_threshold: float = 0.7):
        """
        Semantic chunking based on embedding similarity
        Chunks are created at semantic boundaries
        """
        # This would use embeddings to find natural breakpoints
        # Placeholder for advanced implementation
        pass
    
    @staticmethod
    def hierarchical_chunking(text: str) -> Dict:
        """
        Create hierarchical chunks: document -> section -> paragraph
        Useful for maintaining document structure
        """
        # Placeholder for hierarchical chunking
        # Would preserve headers, sections, etc.
        pass

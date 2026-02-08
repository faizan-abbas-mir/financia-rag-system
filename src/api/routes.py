"""
API Routes for FinanceRAG
Defines all REST API endpoints
"""

import os
import tempfile
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
import anthropic

from utils.document_processor import DocumentProcessor
from utils.metrics import PerformanceTimer


router = APIRouter()


# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    metrics: dict


class UploadResponse(BaseModel):
    success: bool
    filename: str
    chunks_created: int
    message: str


class MetricsResponse(BaseModel):
    total_documents: int
    total_queries: int
    avg_latency_ms: float
    avg_relevance: float


@router.post("/upload", response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Upload and process a document
    
    Args:
        file: Document file (PDF, DOCX, TXT)
        
    Returns:
        Upload status and processing results
    """
    try:
        # Get instances from app state
        vector_store = request.app.state.vector_store
        metrics = request.app.state.metrics
        
        # Validate file type
        allowed_extensions = ['.pdf', '.docx', '.txt']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not supported. Allowed: {allowed_extensions}"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Process document
            with PerformanceTimer() as timer:
                processor = DocumentProcessor(
                    chunk_size=int(os.getenv("CHUNK_SIZE", 512)),
                    chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 50))
                )
                result = processor.process_file(tmp_file_path)
            
            # Add chunks to vector store
            chunk_texts = [chunk['text'] for chunk in result['chunks']]
            chunk_metadatas = [
                {
                    **result['metadata'],
                    'chunk_id': chunk['chunk_id'],
                    'start_idx': chunk['start_idx'],
                    'end_idx': chunk['end_idx']
                }
                for chunk in result['chunks']
            ]
            
            vector_store.add_documents(chunk_texts, chunk_metadatas)
            
            # Record metrics
            metrics.record_document_upload(
                filename=file.filename,
                num_chunks=len(chunk_texts),
                processing_time=timer.elapsed
            )
            
            return UploadResponse(
                success=True,
                filename=file.filename,
                chunks_created=len(chunk_texts),
                message=f"Document processed successfully in {timer.elapsed:.2f}s"
            )
            
        finally:
            # Clean up temp file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: Request, query_request: QueryRequest):
    """
    Query the document collection
    
    Args:
        query_request: Query parameters
        
    Returns:
        Generated answer with sources and metrics
    """
    try:
        # Get instances from app state
        vector_store = request.app.state.vector_store
        metrics = request.app.state.metrics
        
        # Retrieve relevant documents
        with PerformanceTimer() as retrieval_timer:
            results = vector_store.search(query_request.query, top_k=query_request.top_k)
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found. Please upload documents first."
            )
        
        # Build context from retrieved chunks
        context = "\n\n---\n\n".join([r['text'] for r in results])
        
        # Generate answer using Claude
        with PerformanceTimer() as generation_timer:
            answer = await generate_answer(query_request.query, context)
        
        # Extract relevance scores
        relevance_scores = [r['score'] for r in results]
        
        # Record metrics
        metrics.record_query(
            query=query_request.query,
            response=answer,
            retrieval_time=retrieval_timer.elapsed,
            generation_time=generation_timer.elapsed,
            relevance_scores=relevance_scores,
            retrieved_chunks=len(results)
        )
        
        # Format sources for response
        sources = [
            {
                'text': r['text'][:200] + '...' if len(r['text']) > 200 else r['text'],
                'score': round(r['score'] * 100, 1),
                'filename': r['metadata'].get('filename', 'Unknown'),
                'chunk_id': r['metadata'].get('chunk_id', 0)
            }
            for r in results
        ]
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            metrics={
                'retrieval_time_ms': round(retrieval_timer.elapsed * 1000, 1),
                'generation_time_ms': round(generation_timer.elapsed * 1000, 1),
                'total_time_ms': round((retrieval_timer.elapsed + generation_timer.elapsed) * 1000, 1),
                'chunks_retrieved': len(results),
                'avg_relevance_score': round(sum(relevance_scores) / len(relevance_scores) * 100, 1)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(request: Request):
    """
    Get current system metrics
    
    Returns:
        Current performance and usage metrics
    """
    try:
        metrics = request.app.state.metrics
        dashboard_metrics = metrics.get_dashboard_metrics()
        
        return MetricsResponse(**dashboard_metrics)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats(request: Request):
    """
    Get detailed statistics about the vector store
    
    Returns:
        Vector store statistics
    """
    try:
        vector_store = request.app.state.vector_store
        return vector_store.get_collection_stats()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/reset")
async def reset_system(request: Request):
    """
    Reset the system (clear all documents and metrics)
    WARNING: This deletes all data!
    
    Returns:
        Reset confirmation
    """
    try:
        vector_store = request.app.state.vector_store
        metrics = request.app.state.metrics
        
        vector_store.reset()
        metrics.reset()
        
        return {"message": "System reset successfully", "status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def generate_answer(query: str, context: str) -> str:
    """
    Generate answer using Claude API
    
    Args:
        query: User query
        context: Retrieved context from documents
        
    Returns:
        Generated answer
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in environment")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    prompt = f"""You are a financial analyst assistant. Based on the following context from financial documents, please answer the user's question. 

Important guidelines:
1. Only use information from the provided context
2. If the answer is not in the context, clearly state that
3. Cite specific numbers, dates, and facts when available
4. Be concise but comprehensive
5. Use professional financial terminology

Context:
{context}

Question: {query}

Please provide a clear, well-structured answer based solely on the information in the context above."""

    message = client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
        max_tokens=1000,
        temperature=float(os.getenv("TEMPERATURE", 0.7)),
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extract text from response
    answer = ""
    for block in message.content:
        if block.type == "text":
            answer += block.text
    
    return answer

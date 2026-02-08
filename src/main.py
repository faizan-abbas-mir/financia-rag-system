"""
FinanceRAG - Main Application Entry Point
FastAPI application for financial document RAG system
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api.routes import router as api_router
from models.vector_store import VectorStore
from utils.metrics import MetricsCollector

# Load environment variables
load_dotenv()

# Global instances
vector_store = None
metrics = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global vector_store, metrics
    
    # Startup
    print("🚀 Starting FinanceRAG...")
    
    # Initialize vector store
    vector_store = VectorStore(
        persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    
    # Initialize metrics collector
    metrics = MetricsCollector()
    
    # Make available to routes
    app.state.vector_store = vector_store
    app.state.metrics = metrics
    
    print("✅ FinanceRAG initialized successfully!")
    
    yield
    
    # Shutdown
    print("🔄 Shutting down FinanceRAG...")
    if vector_store:
        vector_store.close()
    print("👋 Goodbye!")


# Create FastAPI app
app = FastAPI(
    title="FinanceRAG API",
    description="Retrieval-Augmented Generation for Financial Documents",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

# Include API routes
app.include_router(api_router, prefix="/api")


@app.get("/")
async def root():
    """Serve the main UI"""
    from fastapi import Request
    return templates.TemplateResponse("index.html", {"request": Request})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "vector_store": "connected" if vector_store else "disconnected"
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )

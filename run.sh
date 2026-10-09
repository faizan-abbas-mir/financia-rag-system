#!/bin/bash
# FinanceRAG - Complete startup script
# Starts both FastAPI backend and Streamlit frontend

set -e

echo " Starting FinanceRAG System..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "Please update .env with your API keys and configuration"
        exit 1
    fi
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Start FastAPI backend in the background
echo ""
echo "🔧 Starting FastAPI backend on http://localhost:8000"
echo "=================================================="
cd src
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!
sleep 3

# Start Streamlit frontend
cd ..
echo ""
echo " Starting Streamlit frontend on http://localhost:8501"
echo "=================================================="
streamlit run streamlit_app.py

# Cleanup on exit
trap "kill $FASTAPI_PID" EXIT

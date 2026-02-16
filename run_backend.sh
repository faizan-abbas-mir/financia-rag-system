#!/bin/bash
# FinanceRAG - Backend only startup script
# Use this if you want to run the backend separately

set -e

echo " Starting FinanceRAG Backend Only..."

# Check if .env exists
if [ ! -f .env ]; then
    echo " .env file not found."
    echo "Please create .env with your configuration:"
    echo "  cp .env.example .env"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
echo "Checking dependencies..."
pip install -q -r requirements.txt

# Start FastAPI backend
echo ""
echo " Starting FastAPI backend on http://localhost:8000"
echo "=================================================="
echo "Press Ctrl+C to stop"
echo ""

cd src
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

#!/bin/bash
# FinanceRAG - Streamlit Frontend only startup script
# Use this if the backend is already running elsewhere

set -e

echo "Starting FinanceRAG Streamlit UI..."

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
pip install -q streamlit pandas requests

# Start Streamlit frontend
echo ""
echo "Starting Streamlit frontend on http://localhost:8501"
echo "========================================================="
echo "Backend should be running at http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

streamlit run streamlit_app.py

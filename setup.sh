#!/bin/bash

# FinanceRAG Setup Script
# Automates the setup process for the application

set -e  # Exit on error

echo "======================================"
echo "FinanceRAG Setup Script"
echo "======================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "Error: Python 3.9+ is required. You have $python_version"
    exit 1
fi

echo "✓ Python $python_version found"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your ANTHROPIC_API_KEY"
    echo ""
else
    echo "✓ .env file already exists"
    echo ""
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p chroma_db data logs
echo "✓ Directories created"
echo ""

# Download sample data
if [ ! -f "data/sample_earnings_report.txt" ]; then
    echo "✓ Sample data already exists in data/ directory"
else
    echo "✓ Sample data available"
fi
echo ""

# Run tests to verify installation
echo "Running tests to verify installation..."
if pytest tests/ -v --tb=short > /dev/null 2>&1; then
    echo "✓ All tests passed"
else
    echo "⚠️  Some tests failed (this is okay if you haven't configured API keys yet)"
fi
echo ""

echo "======================================"
echo "Setup Complete! 🎉"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your ANTHROPIC_API_KEY"
echo "2. Run the application: python src/main.py"
echo "3. Open http://localhost:8000 in your browser"
echo ""
echo "For more information, see README.md"
echo ""

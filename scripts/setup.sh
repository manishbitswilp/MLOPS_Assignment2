#!/bin/bash

# Setup script for Cats vs Dogs Classification MLOps Pipeline

set -e

echo "=========================================="
echo "Setting up Cats vs Dogs MLOps Pipeline"
echo "=========================================="

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate || source venv/Scripts/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo "✓ Dependencies installed"

# Initialize DVC
echo ""
echo "Initializing DVC..."
if [ ! -d ".dvc" ]; then
    dvc init
    echo "✓ DVC initialized"
else
    echo "✓ DVC already initialized"
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data/raw data/processed models/performance mlruns notebooks

echo "✓ Directories created"

# Copy environment file
echo ""
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✓ Created .env file from .env.example"
        echo "  Please update .env with your configuration"
    fi
fi

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
echo ""
echo "2. Download the dataset:"
echo "   python src/data/download.py"
echo ""
echo "3. Train the model:"
echo "   python src/models/train.py"
echo ""
echo "4. Start the API:"
echo "   uvicorn src.api.main:app --reload"
echo ""
echo "5. View MLflow experiments:"
echo "   mlflow ui"
echo ""
echo "For more information, see README.md"

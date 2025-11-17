#!/bin/bash

# AI Gameplay Bot - Setup Script
# Automates the complete setup process

set -e  # Exit on error

echo "========================================="
echo "  AI Gameplay Bot - Setup Script"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Check Python version
print_info "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
    print_error "Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi
print_success "Python version: $python_version"

# Create virtual environment
print_info "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip --quiet
print_success "Pip upgraded"

# Install dependencies
print_info "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt --quiet
print_success "Dependencies installed"

# Create directory structure
print_info "Creating directory structure..."
mkdir -p data/raw/gameplay_videos
mkdir -p data/raw/annotations
mkdir -p data/processed/frames
mkdir -p feedback
mkdir -p results
mkdir -p models/neural_network
mkdir -p models/transformer
print_success "Directories created"

# Generate sample data
print_info "Generating sample data..."
python scripts/generate_sample_data.py
print_success "Sample data generated"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_info "Creating .env file..."
    cat > .env << EOL
# AI Gameplay Bot Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
NN_PORT=5000
TRANSFORMER_PORT=5001
CONTROL_PORT=8000
EOL
    print_success ".env file created"
fi

# Run tests to verify installation
print_info "Running tests to verify installation..."
if pytest tests/ -q > /dev/null 2>&1; then
    print_success "Tests passed!"
else
    print_error "Some tests failed. Please check the installation."
fi

echo ""
echo "========================================="
echo -e "${GREEN}  Setup Complete!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Train the models:"
echo "     make train-all"
echo "     (or manually: python models/neural_network/nn_training.py)"
echo ""
echo "  3. Start the services:"
echo "     make run-control"
echo ""
echo "  4. Open the web interface:"
echo "     Open frontend/index.html in your browser"
echo ""
echo "For more information:"
echo "  - Setup guide: SETUP.md"
echo "  - API documentation: API.md"
echo "  - Data format: data/README.md"
echo ""
echo "Quick commands:"
echo "  make help          - Show all available commands"
echo "  make quickstart    - Complete setup with training"
echo "  make test          - Run tests"
echo ""

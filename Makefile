# Makefile for AI Gameplay Bot

.PHONY: help install setup data train test clean run-nn run-transformer run-control run-all stop lint format health-check

# Default target
help:
	@echo "AI Gameplay Bot - Available Commands:"
	@echo ""
	@echo "  make install          - Install Python dependencies"
	@echo "  make setup            - Complete project setup (install + data)"
	@echo "  make health-check     - Run installation and health check"
	@echo "  make data             - Generate sample data"
	@echo "  make train-nn         - Train Neural Network model"
	@echo "  make train-transformer - Train Transformer model"
	@echo "  make train-all        - Train both models"
	@echo "  make test             - Run all tests"
	@echo "  make test-coverage    - Run tests with coverage report"
	@echo "  make run-nn           - Run Neural Network service"
	@echo "  make run-transformer  - Run Transformer service"
	@echo "  make run-control      - Run Control Backend"
	@echo "  make run-all          - Run all services (in background)"
	@echo "  make stop             - Stop all running services"
	@echo "  make lint             - Run code linters"
	@echo "  make format           - Format code with black"
	@echo "  make clean            - Clean temporary files"
	@echo "  make clean-all        - Clean everything (including models and data)"
	@echo ""

# Installation
install:
	@echo "Installing dependencies..."
	pip install --upgrade pip || true
	pip install -r requirements.txt --ignore-installed pyparsing || pip install -r requirements.txt
	@echo "Installation complete!"

# Setup
setup: install
	@echo "Setting up project..."
	mkdir -p data/raw/gameplay_videos data/raw/annotations data/processed/frames feedback results
	mkdir -p models/neural_network models/transformer
	@echo "Setup complete! Run 'make data' to generate sample data."

# Generate sample data
data:
	@echo "Generating sample data..."
	python scripts/generate_sample_data.py
	@echo "Sample data generated!"

# Training
train-nn:
	@echo "Training Neural Network model..."
	python models/neural_network/nn_training.py

train-transformer:
	@echo "Training Transformer model..."
	python models/transformer/transformer_training.py

train-all: train-nn train-transformer
	@echo "All models trained!"

# Testing
test:
	@echo "Running tests..."
	pytest tests/ -v

test-coverage:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=models --cov=scripts --cov=deployment --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/"

# Health check
health-check:
	@echo "Running installation and health check..."
	python test_installation.py

# Running services
run-nn:
	@echo "Starting Neural Network service on port 5000..."
	python deployment/deploy_nn.py

run-transformer:
	@echo "Starting Transformer service on port 5001..."
	python deployment/deploy_transformer.py

run-control:
	@echo "Starting Control Backend on port 8000..."
	@echo "Open frontend/index.html in your browser to access the control panel."
	python deployment/control_backend.py

run-all:
	@echo "Starting all services..."
	@echo "Control Backend will be available at http://localhost:8000"
	@echo "Neural Network API: http://localhost:5000"
	@echo "Transformer API: http://localhost:5001"
	@echo "Frontend: Open frontend/index.html in your browser"
	python deployment/control_backend.py &
	sleep 2
	@echo "All services started! Use 'make stop' to stop them."

# Stop services
stop:
	@echo "Stopping all services..."
	-pkill -f "deploy_nn.py" || true
	-pkill -f "deploy_transformer.py" || true
	-pkill -f "control_backend.py" || true
	@echo "All services stopped."

# Code quality
lint:
	@echo "Running linters..."
	flake8 models scripts deployment --max-line-length=120 --exclude=venv
	pylint models scripts deployment --max-line-length=120 || true

format:
	@echo "Formatting code with black..."
	black models scripts deployment tests --line-length=120

# Cleanup
clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage
	@echo "Cleanup complete!"

clean-all: clean stop
	@echo "Deep cleaning (removing models and generated data)..."
	rm -rf data/processed/*
	rm -rf models/neural_network/*.pth
	rm -rf models/neural_network/*.json
	rm -rf models/transformer/*.pth
	rm -rf models/transformer/*.json
	rm -rf results/*
	rm -rf feedback/*
	@echo "Deep cleanup complete!"

# Quick start
quickstart: setup data
	@echo ""
	@echo "==========================================="
	@echo "  Quick Start Complete!"
	@echo "==========================================="
	@echo ""
	@echo "Next steps:"
	@echo "  1. Train models: make train-all"
	@echo "  2. Start services: make run-control"
	@echo "  3. Open frontend/index.html in your browser"
	@echo ""

# Development mode
dev: setup data
	@echo "Starting development environment..."
	@echo "Training models with reduced epochs for faster testing..."
	python -c "import sys; sys.path.insert(0, 'models/neural_network'); from nn_training import main; main()" || echo "Skipping NN training"
	make run-control

# Production mode
production:
	@echo "Starting in production mode..."
	gunicorn -w 4 -b 0.0.0.0:8000 deployment.control_backend:app --daemon
	@echo "Production server started on http://0.0.0.0:8000"
	@echo "Use 'make stop' to stop the server"

# Makefile for AI Gameplay Bot (uv-enabled, keeps original targets)
# Goals:
# - Keep original commands/targets intact
# - Use uv + .venv by default (Python 3.11 for torch compatibility)
# - DO NOT recreate .venv if it exists (no replace prompt)
# - DO NOT reinstall deps on health-check (only run checks; no sync/install)
# - Keep production target AND keep frontend run target

.PHONY: help install install-dev setup data train-nn train-transformer train-all test test-coverage \
        clean clean-all run-nn run-transformer run-control run-all stop lint format \
        health-check quickstart dev production run-frontend \
        install-uv venv ensure-venv ensure-deps ensure-dev \
        production-stop production-logs

.DEFAULT_GOAL := help

# Use POSIX sh (avoids "/usr/bin/env bash" errors)
SHELL := /bin/sh

# Config
PYTHON_VERSION ?= 3.11
VENV_DIR       ?= .venv

# Production settings
HOST           ?= 0.0.0.0
PORT           ?= 8000
WORKERS        ?= 4
PID_FILE       ?= .gunicorn.pid
LOG_DIR        ?= logs
ACCESS_LOG     ?= $(LOG_DIR)/gunicorn_access.log
ERROR_LOG      ?= $(LOG_DIR)/gunicorn_error.log

# Windows detection
ifeq ($(OS),Windows_NT)
	IS_WINDOWS := 1
	PROD_DAEMON ?= 0
else
	IS_WINDOWS := 0
	PROD_DAEMON ?= 1
endif

# Use uv-run python inside venv
RUN_PYTHON := uv run python

help:
	@echo "AI Gameplay Bot - Available Commands:"
	@echo ""
	@echo "  make install            - Install Python dependencies (production)"
	@echo "  make install-dev        - Install Python dependencies + dev tools"
	@echo "  make setup              - Complete project setup (install + folders)"
	@echo "  make health-check       - Run health check ONLY (no reinstall)"
	@echo "  make data               - Generate sample data"
	@echo "  make train-nn           - Train Neural Network model"
	@echo "  make train-transformer  - Train Transformer model"
	@echo "  make train-all          - Train both models"
	@echo "  make test               - Run all tests"
	@echo "  make test-coverage      - Run tests with coverage report"
	@echo "  make run-nn             - Run Neural Network service"
	@echo "  make run-transformer    - Run Transformer service"
	@echo "  make run-control        - Run Control Backend (dev)"
	@echo "  make run-all            - Run all services (background-ish)"
	@echo "  make run-frontend       - Open frontend/index.html (best-effort)"
	@echo "  make stop               - Stop all running services"
	@echo "  make lint               - Run code linters"
	@echo "  make format             - Format code with black"
	@echo "  make clean              - Clean temporary files"
	@echo "  make clean-all          - Clean everything (including models and data)"
	@echo "  make production         - Run API in production (gunicorn, logs+pid)"
	@echo "  make production-stop    - Stop production gunicorn using pidfile"
	@echo "  make production-logs    - Tail production error log"
	@echo ""

# ---- uv bootstrap (only if missing) ----
install-uv:
	@echo "Checking for uv..."
	@if command -v uv >/dev/null 2>&1; then \
		echo "uv already installed (skipping install)."; \
	else \
		echo "Installing uv package manager..."; \
		if [ "$(IS_WINDOWS)" = "1" ]; then \
			powershell -NoProfile -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"; \
		else \
			if command -v curl >/dev/null 2>&1; then \
				curl -LsSf https://astral.sh/uv/install.sh | sh; \
			elif command -v wget >/dev/null 2>&1; then \
				wget -qO- https://astral.sh/uv/install.sh | sh; \
			else \
				echo "ERROR: uv is missing and neither curl nor wget is available to install it."; \
				echo "Install curl/wget, or install uv using your system package manager."; \
				exit 1; \
			fi; \
		fi; \
	fi

# Create venv ONLY if it doesn't exist (NO replace prompt)
venv: install-uv
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "$(VENV_DIR) already exists (skipping venv creation)."; \
	else \
		echo "Creating virtual environment in $(VENV_DIR) using Python $(PYTHON_VERSION)..."; \
		uv venv --python $(PYTHON_VERSION) $(VENV_DIR); \
		echo "Virtual environment created."; \
	fi

ensure-venv: venv
	@true

# Install prod deps (prefers uv sync if pyproject exists; else requirements.txt)
ensure-deps: ensure-venv
	@if [ -f "pyproject.toml" ]; then \
		echo "Ensuring production dependencies via uv sync..."; \
		uv sync --no-dev; \
	elif [ -f "requirements.txt" ]; then \
		echo "Ensuring production dependencies via requirements.txt..."; \
		uv pip install --upgrade pip; \
		uv pip install -r requirements.txt; \
	else \
		echo "ERROR: No pyproject.toml or requirements.txt found."; \
		exit 1; \
	fi

# Dev deps for pytest/black/flake8/pylint
ensure-dev: ensure-venv
	@if [ -f "pyproject.toml" ]; then \
		echo "Ensuring dev dependencies via uv sync --extra dev..."; \
		uv sync --extra dev; \
	else \
		echo "Ensuring dev tools via requirements-dev.txt (fallback)..."; \
		if [ -f "requirements-dev.txt" ]; then \
			uv pip install -r requirements-dev.txt; \
		else \
			echo "NOTE: No pyproject.toml dev extras and no requirements-dev.txt. Dev tools may be missing."; \
		fi; \
	fi

# ---- Original targets (kept) ----

install: ensure-deps
	@echo "Installation complete!"

install-dev: ensure-dev
	@echo "Dev installation complete!"

setup: install
	@echo "Setting up project..."
	@mkdir -p data/raw/gameplay_videos data/raw/annotations data/processed/frames feedback results
	@mkdir -p models/neural_network models/transformer
	@echo "Setup complete! Run 'make data' to generate sample data."

data: ensure-venv
	@echo "Generating sample data..."
	@$(RUN_PYTHON) scripts/generate_sample_data.py
	@echo "Sample data generated!"

train-nn: ensure-venv
	@echo "Training Neural Network model..."
	@$(RUN_PYTHON) models/neural_network/nn_training.py

train-transformer: ensure-venv
	@echo "Training Transformer model..."
	@$(RUN_PYTHON) models/transformer/transformer_training.py

train-all: train-nn train-transformer
	@echo "All models trained!"

test: ensure-dev
	@echo "Running tests..."
	@uv run pytest tests/ -v

test-coverage: ensure-dev
	@echo "Running tests with coverage..."
	@uv run pytest tests/ --cov=models --cov=scripts --cov=deployment --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/"

# Health check: NO sync/install. Just run the checker inside the existing env.
# If deps are missing, the checker will report it (which is what you want).
health-check: ensure-venv
	@echo "Running installation and health check (no reinstall)..."
	@$(RUN_PYTHON) test_installation.py

run-nn: ensure-venv
	@echo "Starting Neural Network service on port 5000..."
	@$(RUN_PYTHON) deployment/deploy_nn.py

run-transformer: ensure-venv
	@echo "Starting Transformer service on port 5001..."
	@$(RUN_PYTHON) deployment/deploy_transformer.py

run-control: ensure-venv
	@echo "Starting Control Backend on port 8000..."
	@echo "UI: http://localhost:8000/"
	@$(RUN_PYTHON) deployment/control_backend.py

run-all: ensure-venv
	@echo "Starting all services..."
	@echo "Control Backend:  http://localhost:8000"
	@echo "Neural Network:   http://localhost:5000"
	@echo "Transformer:      http://localhost:5001"
	@echo "Frontend:         http://localhost:8000/"
	@$(RUN_PYTHON) deployment/control_backend.py & \
		sleep 1; \
		echo "All services started! Use 'make stop' to stop them."

run-frontend:
	@echo "Frontend is best accessed via backend: http://localhost:8000/"
	@echo "Also opening frontend/index.html directly (best-effort)..."
ifeq ($(IS_WINDOWS),1)
	@powershell -NoProfile -c "Start-Process (Resolve-Path 'frontend/index.html')" || echo "Open frontend/index.html manually."
else
	@if command -v xdg-open >/dev/null 2>&1; then xdg-open frontend/index.html >/dev/null 2>&1 || true; \
	elif command -v open >/dev/null 2>&1; then open frontend/index.html >/dev/null 2>&1 || true; \
	else echo "No opener found. Open frontend/index.html manually."; fi
endif

stop:
	@echo "Stopping all services..."
ifeq ($(IS_WINDOWS),1)
	@powershell -NoProfile -c "Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force" || echo "Stopped (best-effort)."
else
	-pkill -f "deploy_nn.py" || true
	-pkill -f "deploy_transformer.py" || true
	-pkill -f "control_backend.py" || true
	@if [ -f "$(PID_FILE)" ]; then \
		kill $$(cat "$(PID_FILE)") >/dev/null 2>&1 || true; \
		rm -f "$(PID_FILE)" >/dev/null 2>&1 || true; \
	fi
endif
	@echo "All services stopped."

lint: ensure-dev
	@echo "Running linters..."
	@uv run flake8 models scripts deployment tests --max-line-length=120 --exclude=venv
	@uv run pylint models scripts deployment tests --max-line-length=120 || true

format: ensure-dev
	@echo "Formatting code with black..."
	@uv run black models scripts deployment tests --line-length=120

clean:
	@echo "Cleaning temporary files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf htmlcov/ .coverage 2>/dev/null || true
	@echo "Cleanup complete!"

clean-all: clean stop
	@echo "Deep cleaning (removing models and generated data)..."
	@rm -rf data/processed/* 2>/dev/null || true
	@rm -rf models/neural_network/*.pth models/neural_network/*.json 2>/dev/null || true
	@rm -rf models/transformer/*.pth models/transformer/*.json 2>/dev/null || true
	@rm -rf results/* feedback/* 2>/dev/null || true
	@echo "Deep cleanup complete!"

quickstart: setup data
	@echo ""
	@echo "==========================================="
	@echo "  Quick Start Complete!"
	@echo "==========================================="
	@echo ""
	@echo "Next steps:"
	@echo "  1. Train models: make train-all"
	@echo "  2. Start services: make run-control"
	@echo "  3. Open UI: http://localhost:8000/"
	@echo ""

dev: setup data
	@echo "Starting development environment..."
	@echo "Training models with reduced epochs for faster testing..."
	@uv run python -c "import sys; sys.path.insert(0, 'models/neural_network'); from nn_training import main; main()" || echo "Skipping NN training"
	@make run-control

# Production mode (gunicorn)
production: ensure-deps
	@echo "Starting in production mode..."
	@mkdir -p "$(LOG_DIR)"
	@$(RUN_PYTHON) -c "import importlib; importlib.import_module('deployment.control_backend')" || (echo "ERROR: cannot import deployment.control_backend"; exit 1)
	@if [ "$(PROD_DAEMON)" = "1" ]; then \
		uv run gunicorn -w $(WORKERS) -b $(HOST):$(PORT) \
			--pid "$(PID_FILE)" \
			--access-logfile "$(ACCESS_LOG)" \
			--error-logfile "$(ERROR_LOG)" \
			--log-level info \
			--daemon \
			deployment.control_backend:app || (echo "ERROR: gunicorn failed. See $(ERROR_LOG)"; exit 1); \
		sleep 1; \
		if [ ! -f "$(PID_FILE)" ]; then \
			echo "ERROR: gunicorn did not create pidfile. It likely crashed. See $(ERROR_LOG)"; \
			exit 1; \
		fi; \
		echo "Production server started: http://localhost:$(PORT)/"; \
		echo "UI is served by backend now (no 404 on /)."; \
	else \
		uv run gunicorn -w $(WORKERS) -b $(HOST):$(PORT) \
			--access-logfile "-" \
			--error-logfile "-" \
			--log-level info \
			deployment.control_backend:app; \
	fi

production-stop:
	@echo "Stopping production server (pidfile)..."
	@if [ -f "$(PID_FILE)" ]; then \
		kill $$(cat "$(PID_FILE)") >/dev/null 2>&1 || true; \
		rm -f "$(PID_FILE)" >/dev/null 2>&1 || true; \
		echo "Stopped."; \
	else \
		echo "No pidfile found ($(PID_FILE)). Is production running?"; \
	fi

production-logs:
	@echo "Tailing production error log (Ctrl+C to stop)..."
	@if [ -f "$(ERROR_LOG)" ]; then tail -n 200 -f "$(ERROR_LOG)"; else echo "No error log yet at $(ERROR_LOG)."; fi

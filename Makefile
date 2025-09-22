.PHONY: all setup lint fmt test run clean help install-dev install-precommit

# Configuration
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
UVICORN := $(VENV_DIR)/bin/uvicorn
PYTEST := $(VENV_DIR)/bin/pytest
RUFF := $(VENV_DIR)/bin/ruff
BLACK := $(VENV_DIR)/bin/black
PRE_COMMIT := $(VENV_DIR)/bin/pre-commit

# App configuration
APP_MODULE := src/olive/api.py
APP_PATH := olive.api:app

# Default target
all: setup test

# Setup virtual environment and install dependencies
setup: $(VENV_DIR) install-dev install-precommit
	@echo "✅ Setup complete! Virtual environment created and dependencies installed."
	@echo "💡 Run 'make test' to verify everything works."

$(VENV_DIR):
	@echo "Creating virtual environment..."
	python -m venv $(VENV_DIR)
	@echo "Virtual environment created."

install-dev: $(VENV_DIR)
	@echo "📦 Installing development dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo "✅ Development dependencies installed."

install-precommit: $(VENV_DIR)
	@echo "🔧 Installing pre-commit hooks..."
	$(PIP) install pre-commit
	$(PRE_COMMIT) install
	@echo "✅ Pre-commit hooks installed."

# Run linting checks
lint: $(VENV_DIR)
	@echo "🔍 Running linting checks..."
	$(RUFF) check .
	$(RUFF) format --check .
	@echo "✅ Linting checks passed."

# Format code
fmt: $(VENV_DIR)
	@echo "🎨 Formatting code..."
	$(BLACK) .
	$(RUFF) format .
	@echo "✅ Code formatted."

# Run tests with coverage
test: $(VENV_DIR)
	@echo "🧪 Running tests with coverage..."
	$(PYTEST) --cov=src/olive --cov-report=term-missing --cov-report=html --cov-fail-under=80
	@echo "✅ Tests completed."

# Run the FastAPI application
run: $(VENV_DIR)
	@echo "🚀 Starting Olive service..."
	@if [ -f "$(APP_MODULE)" ]; then \
		$(UVICORN) $(APP_PATH) --reload; \
	else \
		echo "Error: FastAPI application file '$(APP_MODULE)' not found."; \
		exit 1; \
	fi

# Clean up generated files
clean:
	@echo "🧹 Cleaning up..."
	rm -rf $(VENV_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .ruff_cache
	rm -rf bandit-report.json
	@echo "✅ Clean up complete."

# Display help message
help:
	@echo "Available targets:"
	@echo "  setup    - Create virtual environment and install dependencies"
	@echo "  lint     - Run code linting checks (ruff + black check)"
	@echo "  fmt      - Format code with black and ruff"
	@echo "  test     - Run tests with coverage"
	@echo "  run      - Start the FastAPI application"
	@echo "  clean    - Remove virtual environment and cache files"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Quick start:"
	@echo "  make setup  # First time setup"
	@echo "  make test   # Run tests"
	@echo "  make run    # Start the service"

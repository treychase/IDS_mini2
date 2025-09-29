.PHONY: help install run test test-verbose lint clean format format-check check-deps setup

help:
	@echo "Available targets:"
	@echo "  install         Install Python dependencies"
	@echo "  setup           Setup development environment (install + check dependencies)"
	@echo "  run             Run the driveline.py script"
	@echo "  test            Run unit tests"
	@echo "  test-verbose    Run tests with verbose output"
	@echo "  test-coverage   Run tests with coverage reporting"
	@echo "  test-debug      Debug test environment"
	@echo "  test-list       List available tests"
	@echo "  test-specific   Run specific test (use TEST=TestName.test_method)"
	@echo "  lint            Lint Python code with flake8"
	@echo "  format          Format code with black"
	@echo "  format-check    Check if code needs formatting (dry-run)"
	@echo "  check-deps      Check if required dependencies are installed"
	@echo "  clean           Remove Python cache files and temporary files"
	@echo "  dev-setup       Complete development environment setup"
	@echo "  dev-check       Run linting and tests"
	@echo "  ci              Full CI pipeline (clean, setup, lint, test)"

install:
	@echo "Installing Python dependencies..."
	pip3 install --user --upgrade pip
	pip3 install --user -r requirements.txt || pip3 install --user pandas matplotlib scikit-learn scipy black flake8 coverage

setup: install check-deps
	@echo "Development environment setup complete!"

run:
	@echo "Running driveline analysis pipeline..."
	python3 driveline.py

test:
	@echo "Running unit tests..."
	python3 test_driveline.py

test-verbose:
	@echo "Running unit tests with verbose output..."
	python3 -m unittest test_driveline -v

test-coverage:
	@echo "Running tests with coverage (requires coverage package)..."
	@which coverage > /dev/null || (echo "Installing coverage..." && pip3 install --user coverage)
	coverage run test_driveline.py
	coverage report -m
	coverage html

lint:
	@echo "Linting Python code..."
	@which flake8 > /dev/null || (echo "Installing flake8..." && pip3 install --user flake8)
	flake8 driveline.py --max-line-length=100 --ignore=E501,W503,E203 || echo "Linting completed with warnings"
	flake8 test_driveline.py --max-line-length=120 --ignore=E501,W503,E203 || echo "Test linting completed with warnings"

format:
	@echo "Formatting Python code with black..."
	@which black > /dev/null || (echo "Installing black..." && pip3 install --user black)
	black driveline.py test_driveline.py
	@echo "Code formatting complete!"

format-check:
	@echo "Checking code formatting with black (dry-run)..."
	@which black > /dev/null || (echo "Installing black..." && pip3 install --user black)
	black --check --diff driveline.py test_driveline.py

check-deps:
	@echo "Checking required dependencies..."
	@python3 -c "import pandas; print('✓ pandas')" || echo "✗ pandas (missing)"
	@python3 -c "import matplotlib; print('✓ matplotlib')" || echo "✗ matplotlib (missing)"
	@python3 -c "import sklearn; print('✓ scikit-learn')" || echo "✗ scikit-learn (missing)"
	@python3 -c "import numpy; print('✓ numpy')" || echo "✗ numpy (missing)"
	@python3 -c "import scipy; print('✓ scipy')" || echo "✗ scipy (missing - needed for z-score outlier detection)"
	@python3 -c "import black; print('✓ black')" || echo "✗ black (missing - needed for code formatting)"

validate-data:
	@echo "Validating data file exists..."
	@test -f hp_obp.csv && echo "✓ hp_obp.csv found" || echo "✗ hp_obp.csv not found - pipeline will fail"

run-full: validate-data run

test-quick:
	@echo "Running quick smoke tests..."
	python3 -c "import driveline; print('✓ Module imports successfully')"
	python3 -c "from driveline import load_data, run_complete_pipeline; print('✓ Main functions importable')"

clean:
	@echo "Cleaning up Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleanup complete!"

# Development workflow shortcuts
dev-setup: setup test
	@echo "Development setup complete and tests passing!"

dev-check: lint format-check test
	@echo "Development checks complete!"

# CI/CD style checks
ci: clean setup lint format-check test
	@echo "CI pipeline complete!"

# Help for specific commands
help-test:
	@echo "Test commands:"
	@echo "  test         - Run all unit tests"
	@echo "  test-verbose - Run tests with detailed output"
	@echo "  test-coverage- Run tests with coverage report"
	@echo "  test-quick   - Quick smoke test for imports"

help-dev:
	@echo "Development commands:"
	@echo "  dev-setup    - Complete development environment setup"
	@echo "  dev-check    - Run linting, format check, and tests"
	@echo "  format       - Auto-format code with black"
	@echo "  format-check - Check if code needs formatting"
	@echo "  lint         - Check code style with flake8"
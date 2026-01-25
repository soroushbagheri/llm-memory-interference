.PHONY: install setup test lint format clean run-demo run-experiments docs

# Setup
install:
	pip install -e .

setup:
	pip install -r requirements.txt
	python -m pip install -e .
	mkdir -p data/raw data/processed data/cache
	mkdir -p results/experiments results/plots results/logs
	mkdir -p models/checkpoints

setup-dev: setup
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

# Code Quality
lint:
	flake8 src/ tests/ --max-line-length=100
	mypy src/

format:
	black src/ tests/ experiments/

format-check:
	black --check src/ tests/ experiments/

# Running
run-demo:
	python experiments/demo.py

run-detector:
	python experiments/run_detector.py --config configs/default.yaml

run-experiments:
	python experiments/run_benchmark.py --config configs/benchmark.yaml

run-analysis:
	python experiments/analysis.py --results results/experiments/

# Data
generate-dataset:
	python scripts/generate_interference_dataset.py --output data/processed/

# Visualization
generate-plots:
	python scripts/generate_plots.py --results results/experiments/ --output results/plots/

# Documentation
docs:
	mkdir -p docs/_build
	cd docs && make html

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache .coverage htmlcov/ .mypy_cache/

clean-data:
	rm -rf data/cache/*
	rm -rf data/processed/*

clean-results:
	rm -rf results/experiments/*
	rm -rf results/plots/*

clean-all: clean clean-data clean-results
	 rm -rf build/ dist/ *.egg-info/

# Help
help:
	@echo "Available commands:"
	@echo "  make setup          - Install dependencies and create directories"
	@echo "  make test           - Run tests"
	@echo "  make lint           - Check code quality"
	@echo "  make format         - Format code with black"
	@echo "  make run-demo       - Run quick demonstration"
	@echo "  make run-experiments - Run full benchmark experiments"
	@echo "  make generate-dataset - Generate interference dataset"
	@echo "  make generate-plots - Generate visualizations"
	@echo "  make clean          - Remove temporary files"

.PHONY: help install test lint format type-check clean ci

PYTHON := python3
TOX := tox

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install the package and development dependencies"
	@echo "  test        - Run tests using pytest"
	@echo "  lint        - Run linting checks with Ruff"
	@echo "  format      - Auto-format code with Ruff"
	@echo "  type-check  - Run type checking"
	@echo "  clean       - Remove build artifacts and cache files"
	@echo "  ci          - Run all checks (test, lint, format, type-check)"
	@echo "  tox         - Run tests with tox"
	@echo "  tox-test    - Run tests with tox"
	@echo "  tox-lint    - Run linting with tox"
	@echo "  build       - Build the package"
	@echo "  upload-pypi - Upload the package to PyPI"
	@echo "  release     - Clean, build, and upload the package to PyPI"


install:
	$(PYTHON) -m pip install -e .[dev]
	pre-commit install

test:
	$(PYTHON) -m pytest tests --asyncio-mode=auto

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

type-check:
	$(PYTHON) -m mypy .

clean:
	rm -rf build dist .eggs *.egg-info
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	$(PYTHON) -c "import shutil; import os; [shutil.rmtree(os.path.join(root, d)) for root, dirs, _ in os.walk('.') for d in dirs if d == '__pycache__']"

ci:
	$(TOX) -e py310,py311,py312,lint-format-type

tox:
	$(TOX)

tox-test:
	$(TOX) -e py310,py311,py312

tox-lint:
	$(TOX) -e lint-format-type

build:
	python -m build

upload-pypi:
	python -m twine upload dist/*

release: clean build upload-pypi
	@echo "Released new version to PyPI"

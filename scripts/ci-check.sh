#!/bin/bash
# Quick local CI check - runs the same checks as GitHub Actions
# Usage: ./scripts/ci-check.sh

set -e  # Exit on error

echo "🔍 Running CI checks locally..."
echo ""

echo "→ Linting with ruff..."
poetry run ruff check src/ tests/
echo "✓ Ruff passed"
echo ""

echo "→ Checking formatting with black..."
poetry run black --check src/ tests/
echo "✓ Black passed"
echo ""

echo "→ Type checking with mypy..."
poetry run mypy src/myriad/
echo "✓ MyPy passed"
echo ""

echo "→ Running tests..."
poetry run pytest tests/ --ignore=tests/examples -v
echo "✓ Tests passed"
echo ""

echo "→ Running example tests..."
poetry run pytest tests/examples/ -m "not slow" -v
echo "✓ Example tests passed"
echo ""

echo "✅ All CI checks passed! Safe to push."

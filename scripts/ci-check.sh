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

echo "→ Checking tutorial notebooks have pre-executed outputs..."
failed=0
for notebook in $(find docs/tutorials -name "*.ipynb" -type f); do
  output_count=$(python -c "import json; nb=json.load(open('$notebook')); print(sum(len(cell.get('outputs', [])) for cell in nb['cells']))")
  if [ "$output_count" -eq 0 ]; then
    echo "  ❌ $notebook has no outputs (run: jupyter nbconvert --to notebook --execute $notebook --inplace)"
    failed=1
  else
    echo "  ✓ $notebook"
  fi
done
[ $failed -eq 1 ] && exit 1
echo "✓ All notebooks have pre-executed outputs"
echo ""

echo "→ Checking for broken literalinclude references..."
broken=0
while IFS= read -r line; do
  example=$(echo "$line" | sed 's/.*examples\/\([^[:space:]]*\).*/\1/')
  if [ ! -f "examples/$example" ]; then
    echo "  ❌ Missing examples/$example"
    broken=1
  fi
done < <(grep -r "literalinclude.*examples/" docs/ 2>/dev/null || true)
[ $broken -eq 1 ] && exit 1
echo "✓ All example files exist"
echo ""

echo "→ Building Sphinx docs..."
poetry run sphinx-build -W --keep-going -b html docs/ docs/_build/html
echo "✓ Docs build passed"
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

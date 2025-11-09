#!/bin/bash
set -e

echo "🔧 Auto-fixing linting issues in all files (C++/CUDA + Python)..."
echo ""

# Fix C++/CUDA files
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "scripts/lint/cpp-fix.sh" ]; then
    scripts/lint/cpp-fix.sh
    CPP_EXIT=$?
else
    echo "⚠️  C++/CUDA lint-fix script not found"
    CPP_EXIT=0
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Fix Python files
if [ -f "scripts/lint/python-fix.sh" ]; then
    scripts/lint/python-fix.sh
    PYTHON_EXIT=$?
else
    echo "⚠️  Python lint-fix script not found"
    PYTHON_EXIT=0
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Auto-fix complete!"

if [ $CPP_EXIT -ne 0 ] || [ $PYTHON_EXIT -ne 0 ]; then
    exit 1
fi


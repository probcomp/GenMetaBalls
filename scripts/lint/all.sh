#!/bin/bash
set -e

echo "🔍 Linting all files (C++/CUDA + Python)..."
echo ""

# Lint C++/CUDA files
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "scripts/lint/cpp.sh" ]; then
    scripts/lint/cpp.sh
    CPP_EXIT=$?
else
    echo "⚠️  C++/CUDA lint script not found"
    CPP_EXIT=0
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Lint Python files
if [ -f "scripts/lint/python.sh" ]; then
    scripts/lint/python.sh
    PYTHON_EXIT=$?
else
    echo "⚠️  Python lint script not found"
    PYTHON_EXIT=0
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $CPP_EXIT -eq 0 ] && [ $PYTHON_EXIT -eq 0 ]; then
    echo "✨ All files passed linting!"
    exit 0
else
    echo "❌ Found linting issues."
    exit 1
fi


#!/bin/bash
set -e

echo "🔍 Checking formatting for all files (C++/CUDA + Python)..."
echo ""

# Check C++/CUDA files
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "scripts/format/cpp.sh" ]; then
    scripts/format/cpp.sh check
    CPP_EXIT=$?
else
    echo "⚠️  C++/CUDA format script not found"
    CPP_EXIT=0
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check Python files
if [ -f "scripts/format/python.sh" ]; then
    scripts/format/python.sh check
    PYTHON_EXIT=$?
else
    echo "⚠️  Python format script not found"
    PYTHON_EXIT=0
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $CPP_EXIT -eq 0 ] && [ $PYTHON_EXIT -eq 0 ]; then
    echo "✨ All files are properly formatted!"
    exit 0
else
    echo "❌ Some files need formatting. Run 'pixi run format' to fix."
    exit 1
fi


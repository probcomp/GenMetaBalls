#!/bin/bash
set -e

echo "🔧 Formatting all files (C++/CUDA + Python)..."
echo ""

# Format C++/CUDA files
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "scripts/format/cpp.sh" ]; then
    scripts/format/cpp.sh
    CPP_EXIT=$?
else
    echo "⚠️  C++/CUDA format script not found"
    CPP_EXIT=0
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Format Python files
if [ -f "scripts/format/python.sh" ]; then
    scripts/format/python.sh
    PYTHON_EXIT=$?
else
    echo "⚠️  Python format script not found"
    PYTHON_EXIT=0
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Formatting complete!"

if [ $CPP_EXIT -ne 0 ] || [ $PYTHON_EXIT -ne 0 ]; then
    exit 1
fi


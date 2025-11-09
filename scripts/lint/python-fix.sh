#!/bin/bash
set -e

PYTHON_FILES=$(find genmetaballs tests -type f -name '*.py' 2>/dev/null | sort || true)

if [ -z "$PYTHON_FILES" ]; then
    echo "✓ No Python files found"
    exit 0
fi

echo "🔧 Auto-fixing linting issues in Python files..."
FIXED_COUNT=0
ISSUE_COUNT=0

# Run ruff check with --fix
RUFF_OUTPUT=$(ruff check --fix $PYTHON_FILES 2>&1 || true)
RUFF_EXIT=$?

# Process each file to show what was fixed
while IFS= read -r file; do
    if [ -f "$file" ]; then
        # Check if this file had any issues
        FILE_ISSUES=$(echo "$RUFF_OUTPUT" | grep "^$file" || true)
        if [ -z "$FILE_ISSUES" ]; then
            echo "  ✓ $file (no issues)"
        else
            echo "  ✗ $file (fixed issues)"
            echo "$FILE_ISSUES" | sed 's/^/    /'
            ISSUE_COUNT=$((ISSUE_COUNT + 1))
        fi
        FIXED_COUNT=$((FIXED_COUNT + 1))
    fi
done <<< "$PYTHON_FILES"

echo ""
if [ $RUFF_EXIT -eq 0 ] && [ $ISSUE_COUNT -eq 0 ]; then
    echo "✨ All files are clean! ($FIXED_COUNT files checked)"
else
    echo "✨ Auto-fix complete! ($ISSUE_COUNT files had fixes applied)"
    echo "💡 Note: Some issues may require manual fixes. Run 'pixi run lint-python' to check remaining issues"
fi


#!/bin/bash
set -e

PYTHON_FILES=$(find genmetaballs tests -type f -name '*.py' 2>/dev/null | sort || true)

if [ -z "$PYTHON_FILES" ]; then
    echo "✓ No Python files found"
    exit 0
fi

echo "🔍 Linting Python files..."
HAS_ISSUES=0
CLEAN_COUNT=0
ISSUE_COUNT=0

# Run ruff check and capture output
set +e
RUFF_OUTPUT=$(ruff check $PYTHON_FILES 2>&1)
RUFF_EXIT=$?
set -e

# If ruff found issues, show them grouped by file
if [ $RUFF_EXIT -ne 0 ] && [ -n "$RUFF_OUTPUT" ]; then
    # Process each file
    OLD_IFS="$IFS"
    IFS=$'\n'
    FILES_WITH_ISSUES=""
    for file in $PYTHON_FILES; do
        IFS="$OLD_IFS"
        if [ -f "$file" ]; then
            # Check if this file appears in the error output
            if echo "$RUFF_OUTPUT" | grep -q "$file"; then
                echo "  ✗ $file"
                # Extract error blocks for this file - from error code through help line
                echo "$RUFF_OUTPUT" | python3 -c "
import sys
file = '$file'
lines = sys.stdin.readlines()
in_block = False
block_lines = []
for i, line in enumerate(lines):
    # Start of error block for our file
    if file in line and '-->' in line:
        # Print previous block if any
        if block_lines:
            for bl in block_lines:
                print('    ' + bl.rstrip())
            block_lines = []
        in_block = True
        # Find the error code line before this (look back up to 2 lines)
        for j in range(max(0, i-2), i):
            if j < len(lines) and lines[j].strip():
                stripped = lines[j].strip()
                if len(stripped) >= 2 and stripped[0].isupper() and stripped[1].isdigit():
                    block_lines.append(lines[j])
                    break
        block_lines.append(line)
    elif in_block:
        block_lines.append(line)
        # End of block at help: line or next error code or Found
        if 'help:' in line or (line.strip() and line[0].isupper() and line[1].isdigit() and file not in line):
            in_block = False
            for bl in block_lines:
                print('    ' + bl.rstrip())
            block_lines = []
        elif 'Found' in line:
            in_block = False
            if block_lines:
                for bl in block_lines:
                    print('    ' + bl.rstrip())
                block_lines = []
if block_lines:
    for bl in block_lines:
        print('    ' + bl.rstrip())
"
                FILES_WITH_ISSUES="$FILES_WITH_ISSUES $file"
                ISSUE_COUNT=$((ISSUE_COUNT + 1))
                HAS_ISSUES=1
            else
                echo "  ✓ $file"
                CLEAN_COUNT=$((CLEAN_COUNT + 1))
            fi
        fi
    done
    IFS="$OLD_IFS"
else
    # All files are clean
    OLD_IFS="$IFS"
    IFS=$'\n'
    for file in $PYTHON_FILES; do
        IFS="$OLD_IFS"
        if [ -f "$file" ]; then
            echo "  ✓ $file"
            CLEAN_COUNT=$((CLEAN_COUNT + 1))
        fi
    done
    IFS="$OLD_IFS"
fi

echo ""
if [ $RUFF_EXIT -eq 0 ] && [ $HAS_ISSUES -eq 0 ]; then
    TOTAL_FILES=$(echo "$PYTHON_FILES" | wc -l)
    echo "✨ All Python files passed linting! ($TOTAL_FILES files checked)"
    exit 0
else
    echo "❌ Found issues in $ISSUE_COUNT file(s). ($CLEAN_COUNT files clean)"
    echo "💡 Tip: Run 'ruff check --fix' to auto-fix some issues"
    exit 1
fi


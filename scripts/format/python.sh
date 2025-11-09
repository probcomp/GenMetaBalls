#!/bin/bash
set -e

PYTHON_FILES=$(find genmetaballs tests -type f -name '*.py' 2>/dev/null | sort || true)

if [ -z "$PYTHON_FILES" ]; then
    echo "✓ No Python files found"
    exit 0
fi

if [ "$1" = "check" ]; then
    echo "🔍 Checking Python file formatting..."
    NEEDS_FORMAT=0
    
    # First check import ordering
    ruff check --select I --fix --quiet $PYTHON_FILES >/dev/null 2>&1 || true
    
    # Then check formatting
    OLD_IFS="$IFS"
    IFS=$'\n'
    for file in $PYTHON_FILES; do
        IFS="$OLD_IFS"
        if [ -f "$file" ]; then
            set +e
            ERROR_OUTPUT=$(ruff format --check "$file" 2>&1)
            EXIT_CODE=$?
            set -e
            if [ $EXIT_CODE -eq 0 ]; then
                echo "  ✓ $file"
            else
                echo "  ✗ $file needs formatting"
                if [ -n "$ERROR_OUTPUT" ]; then
                    echo "$ERROR_OUTPUT" | sed 's/^/    /'
                else
                    echo "    File does not match ruff format style"
                fi
                NEEDS_FORMAT=1
            fi
        fi
    done
    IFS="$OLD_IFS"
    
    if [ $NEEDS_FORMAT -eq 0 ]; then
        echo "✨ All Python files are properly formatted!"
        exit 0
    else
        echo "❌ Some Python files need formatting. Run 'pixi run format' to fix."
        exit 1
    fi
else
    echo "🔧 Formatting Python files..."
    FORMATTED=0
    ALREADY_FORMATTED=0
    
    OLD_IFS="$IFS"
    IFS=$'\n'
    for file in $PYTHON_FILES; do
        IFS="$OLD_IFS"
        if [ -f "$file" ]; then
            # Check if file needs formatting
            set +e
            ruff format --check "$file" >/dev/null 2>&1
            NEEDS_FORMAT=$?
            set -e
            
            # Fix import ordering first (this might change the file)
            ruff check --select I --fix --quiet "$file" >/dev/null 2>&1 || true
            
            # Then format if needed
            if [ $NEEDS_FORMAT -ne 0 ]; then
                set +e
                ERROR_OUTPUT=$(ruff format "$file" 2>&1)
                EXIT_CODE=$?
                set -e
                if [ $EXIT_CODE -eq 0 ]; then
                    echo "  ✓ Formatted: $file"
                    FORMATTED=1
                else
                    echo "  ✗ Failed to format: $file"
                    # Show full error output with file and line details
                    if [ -n "$ERROR_OUTPUT" ]; then
                        echo "$ERROR_OUTPUT" | sed 's/^/    /'
                    else
                        echo "    Unknown formatting error"
                    fi
                    echo ""
                    echo "💡 Review the error above and fix the issues, then run 'pixi run format' again"
                    exit 1
                fi
            else
                echo "  ✓ Already formatted: $file"
                ALREADY_FORMATTED=1
            fi
        fi
    done
    IFS="$OLD_IFS"
    
    if [ $FORMATTED -eq 1 ]; then
        echo "✨ Python formatting complete!"
    elif [ $ALREADY_FORMATTED -eq 1 ]; then
        echo "✨ All Python files were already formatted!"
    fi
fi


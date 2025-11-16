#!/bin/bash
set -e

echo "🔧 Formatting all files (C++/CUDA + Python)..."
echo ""

# Format C++/CUDA files
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
CPP_FILES=$(find genmetaballs/src/cuda tests -type f \( -name '*.cu' -o -name '*.cuh' \) 2>/dev/null || true)

if [ -z "$CPP_FILES" ]; then
    echo "✓ No CUDA/C++ files found"
    CPP_EXIT=0
else
    echo "🔧 Formatting CUDA/C++ files..."
    FORMATTED=0
    ALREADY_FORMATTED=0
    OLD_IFS="$IFS"
    IFS=$'\n'
    for file in $CPP_FILES; do
        IFS="$OLD_IFS"
        if [ -f "$file" ]; then
            # Check if file needs formatting
            set +e
            clang-format --dry-run --Werror "$file" >/dev/null 2>&1
            CHECK_EXIT=$?
            set -e
            if [ $CHECK_EXIT -eq 0 ]; then
                echo "  ✓ Already formatted: $file"
                ALREADY_FORMATTED=1
            else
                set +e
                ERROR_OUTPUT=$(clang-format -i "$file" 2>&1)
                EXIT_CODE=$?
                set -e
                if [ $EXIT_CODE -eq 0 ]; then
                    echo "  ✓ Formatted: $file"
                    FORMATTED=1
                else
                    echo "  ✗ Failed to format: $file"
                    if [ -n "$ERROR_OUTPUT" ]; then
                        echo "$ERROR_OUTPUT" | sed 's/^/    /'
                    else
                        echo "    Unknown formatting error"
                    fi
                    echo ""
                    echo "💡 Review the error above and fix the issues, then run 'pixi run format' again"
                    exit 1
                fi
            fi
        fi
    done
    IFS="$OLD_IFS"
    
    if [ $FORMATTED -eq 1 ]; then
        echo "✨ C++/CUDA formatting complete!"
    elif [ $ALREADY_FORMATTED -eq 1 ]; then
        echo "✨ All CUDA/C++ files were already formatted!"
    fi
    CPP_EXIT=0
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Format Python files
PYTHON_FILES=$(find genmetaballs tests -type f -name '*.py' 2>/dev/null | sort || true)

if [ -z "$PYTHON_FILES" ]; then
    echo "✓ No Python files found"
    PYTHON_EXIT=0
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
    PYTHON_EXIT=0
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Formatting complete!"

if [ ${CPP_EXIT:-0} -ne 0 ] || [ ${PYTHON_EXIT:-0} -ne 0 ]; then
    exit 1
fi


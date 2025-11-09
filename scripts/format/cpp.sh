#!/bin/bash
set -e

FILES=$(find genmetaballs/src/cuda tests -type f \( -name '*.cu' -o -name '*.cuh' \) 2>/dev/null || true)

if [ -z "$FILES" ]; then
    echo "✓ No CUDA/C++ files found"
    exit 0
fi

if [ "$1" = "check" ]; then
    echo "🔍 Checking CUDA/C++ file formatting..."
    NEEDS_FORMAT=0
    OLD_IFS="$IFS"
    IFS=$'\n'
    for file in $FILES; do
        IFS="$OLD_IFS"
        if [ -f "$file" ]; then
            # Check formatting: capture both stdout and stderr for detailed error messages
            set +e  # Temporarily disable exit on error to capture exit code
            ERROR_OUTPUT=$(clang-format --dry-run --Werror "$file" 2>&1)
            EXIT_CODE=$?
            set -e  # Re-enable exit on error
            if [ $EXIT_CODE -eq 0 ]; then
                echo "  ✓ $file"
            else
                echo "  ✗ $file needs formatting"
                # Show full error output with file and line details
                if [ -n "$ERROR_OUTPUT" ]; then
                    echo "$ERROR_OUTPUT" | sed 's/^/    /'
                else
                    echo "    File does not match clang-format style"
                fi
                NEEDS_FORMAT=1
            fi
        fi
    done
    IFS="$OLD_IFS"
    
    if [ $NEEDS_FORMAT -eq 0 ]; then
        echo "✨ All CUDA/C++ files are properly formatted!"
        exit 0
    else
        echo "❌ Some files need formatting. Run 'pixi run format' to fix."
        exit 1
    fi
else
    echo "🔧 Formatting CUDA/C++ files..."
    FORMATTED=0
    ALREADY_FORMATTED=0
    OLD_IFS="$IFS"
    IFS=$'\n'
    for file in $FILES; do
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
            fi
        fi
    done
    IFS="$OLD_IFS"
    
    if [ $FORMATTED -eq 1 ]; then
        echo "✨ Formatting complete!"
    elif [ $ALREADY_FORMATTED -eq 1 ]; then
        echo "✨ All CUDA/C++ files were already formatted!"
    fi
fi


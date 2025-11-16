#!/bin/bash
set -e

echo "🔧 Auto-fixing linting issues and formatting all files (C++/CUDA + Python)..."
echo ""

# Fix C++/CUDA files
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
CPP_FILES=$(find genmetaballs/src/cuda tests -type f \( -name '*.cu' -o -name '*.cuh' \) 2>/dev/null || true)

if [ -z "$CPP_FILES" ]; then
    echo "✓ No CUDA/C++ files found"
    CPP_EXIT=0
else
    USE_COMPILE_COMMANDS=false
    
    # Try to use compile_commands.json if available
    if [ -f "build/compile_commands.json" ]; then
        USE_COMPILE_COMMANDS=true
    else
        echo "⚠️  compile_commands.json not found. Running 'pixi run compile-commands' first..."
        cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON >/dev/null 2>&1 || true
        if [ -f "build/compile_commands.json" ]; then
            USE_COMPILE_COMMANDS=true
        fi
    fi
    
    echo "🔧 Auto-fixing linting issues in CUDA/C++ files..."
    FIXED=0
    
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            if [ "$USE_COMPILE_COMMANDS" = true ]; then
                # Use compile_commands.json for better analysis
                OUTPUT=$(clang-tidy --fix "$file" 2>&1 || true)
            else
                # Use manual flags
                OUTPUT=$(clang-tidy --fix "$file" -- -Igenmetaballs/src/cuda -std=c++20 2>&1 || true)
            fi
            
            # Check if any fixes were applied (clang-tidy modifies files in place)
            echo "  ✓ Processed: $file"
            FIXED=1
        fi
    done <<< "$CPP_FILES"
    
    echo "✨ C++/CUDA auto-fix complete! ($(echo "$CPP_FILES" | wc -l) files processed)"
    CPP_EXIT=0
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Fix Python files
PYTHON_FILES=$(find genmetaballs tests -type f -name '*.py' 2>/dev/null | sort || true)

if [ -z "$PYTHON_FILES" ]; then
    echo "✓ No Python files found"
    PYTHON_EXIT=0
else
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
    
    if [ $RUFF_EXIT -eq 0 ] && [ $ISSUE_COUNT -eq 0 ]; then
        echo "✨ All Python files are clean! ($FIXED_COUNT files checked)"
    else
        echo "✨ Python auto-fix complete! ($ISSUE_COUNT files had fixes applied)"
    fi
    PYTHON_EXIT=0
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Auto-fix complete! Now formatting files..."
echo ""

# Now format all files
exec scripts/format.sh


#!/bin/bash
set -e

echo "🔍 Linting all files (C++/CUDA + Python)..."
echo ""

# Lint C++/CUDA files
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
CPP_FILES=$(find genmetaballs/src/cuda tests -type f \( -name '*.cu' -o -name '*.cuh' \) 2>/dev/null || true)

if [ -z "$CPP_FILES" ]; then
    echo "✓ No CUDA/C++ files found"
    CPP_EXIT=0
else
    USE_COMPILE_COMMANDS=true
    
    # Try to use compile_commands.json if available (faster and more accurate)
    if [ ! -f "build/compile_commands.json" ]; then
        echo "⚠️  compile_commands.json not found. Generating it..."
        cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON >/dev/null 2>&1 || true
        if [ ! -f "build/compile_commands.json" ]; then
            echo "⚠️  Could not generate compile_commands.json, falling back to basic mode"
            USE_COMPILE_COMMANDS=false
        fi
    fi
    
    echo "🔍 Linting CUDA/C++ files..."
    HAS_ISSUES=0
    CLEAN_FILES=0
    FILES_WITH_ISSUES=0
    
    # Create temporary directory for parallel processing results
    TMPDIR=$(mktemp -d)
    trap "rm -rf $TMPDIR" EXIT
    
    # Export variables for parallel execution
    export USE_COMPILE_COMMANDS
    export TMPDIR
    
    # Run clang-tidy in parallel (use number of CPU cores)
    NPROC=$(nproc 2>/dev/null || echo 4)
    echo "$CPP_FILES" | xargs -P "$NPROC" -I {} sh -c '
        file="$1"
        use_compile_commands="$2"
        tmpdir="$3"
        
        if [ ! -f "$file" ]; then
            exit 0
        fi
        
        if [ "$use_compile_commands" = "true" ]; then
            OUTPUT=$(clang-tidy "$file" 2>&1 || true)
        else
            OUTPUT=$(clang-tidy "$file" -- -Igenmetaballs/src/cuda -std=c++20 2>&1 || true)
        fi
        
        FILE_BASENAME=$(basename "$file")
        ISSUES=$(echo "$OUTPUT" | grep -E "(^|/)$FILE_BASENAME:" | \
            grep -E ": (warning|error):" | \
            grep -v "suppressed" | \
            grep -v "clang-diagnostic-error" | \
            grep -v "__clang_cuda_runtime_wrapper.h" | \
            grep -v "CUDA version is newer" | \
            grep -v "unable to handle compilation" | \
            grep -v "unused-command-line-argument" | \
            grep -v "linker.*input unused" | \
            grep -v "Error parsing.*clang-tidy" | \
            grep -v "Error while processing" | \
            grep -v "Found compiler error" || true)
        
        tmpfile="$tmpdir/$(basename "$file" | tr "/" "_").lint"
        if [ -z "$ISSUES" ]; then
            echo "CLEAN:$file" > "$tmpfile"
        else
            echo "ISSUES:$file" > "$tmpfile"
            echo "$ISSUES" >> "$tmpfile"
        fi
    ' _ {} "$USE_COMPILE_COMMANDS" "$TMPDIR"
    
    # Process results in order
    OLD_IFS="$IFS"
    IFS=$'\n'
    for file in $CPP_FILES; do
        IFS="$OLD_IFS"
        if [ -f "$file" ]; then
            tmpfile="$TMPDIR/$(basename "$file" | tr "/" "_").lint"
            if [ -f "$tmpfile" ]; then
                FIRST_LINE=$(head -n 1 "$tmpfile")
                if echo "$FIRST_LINE" | grep -q "^CLEAN:"; then
                    echo "  ✓ $file"
                    CLEAN_FILES=$((CLEAN_FILES + 1))
                elif echo "$FIRST_LINE" | grep -q "^ISSUES:"; then
                    echo "  ✗ $file"
                    # Show issues with proper indentation
                    tail -n +2 "$tmpfile" | while IFS= read -r line; do
                        if echo "$line" | grep -qE ":[0-9]+:[0-9]+: (warning|error):"; then
                            echo "$line" | sed -E 's/^[^:]+:([0-9]+:[0-9]+: (warning|error):.*)/    \1/'
                        else
                            echo "    $line"
                        fi
                    done
                    FILES_WITH_ISSUES=$((FILES_WITH_ISSUES + 1))
                    HAS_ISSUES=1
                fi
            fi
        fi
    done
    IFS="$OLD_IFS"
    
    if [ $HAS_ISSUES -eq 0 ]; then
        echo "✨ All C++/CUDA files passed linting! ($CLEAN_FILES files checked)"
        CPP_EXIT=0
    else
        echo "❌ Found issues in $FILES_WITH_ISSUES file(s). ($CLEAN_FILES files clean)"
        CPP_EXIT=1
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Lint Python files
PYTHON_FILES=$(find genmetaballs tests -type f -name '*.py' 2>/dev/null | sort || true)

if [ -z "$PYTHON_FILES" ]; then
    echo "✓ No Python files found"
    PYTHON_EXIT=0
else
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
    
    if [ $RUFF_EXIT -eq 0 ] && [ $HAS_ISSUES -eq 0 ]; then
        TOTAL_FILES=$(echo "$PYTHON_FILES" | wc -l)
        echo "✨ All Python files passed linting! ($TOTAL_FILES files checked)"
        PYTHON_EXIT=0
    else
        echo "❌ Found issues in $ISSUE_COUNT file(s). ($CLEAN_COUNT files clean)"
        PYTHON_EXIT=1
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ${CPP_EXIT:-0} -eq 0 ] && [ ${PYTHON_EXIT:-0} -eq 0 ]; then
    echo "✨ All files passed linting!"
    exit 0
else
    echo "❌ Found linting issues."
    exit 1
fi


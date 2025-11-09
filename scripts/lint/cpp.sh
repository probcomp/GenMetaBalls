#!/bin/bash
set -e

FILES=$(find genmetaballs/src/cuda tests -type f \( -name '*.cu' -o -name '*.cuh' \) 2>/dev/null || true)

if [ -z "$FILES" ]; then
    echo "✓ No CUDA/C++ files found"
    exit 0
fi

USE_COMPILE_COMMANDS=true
if [ "$1" = "basic" ]; then
    USE_COMPILE_COMMANDS=false
fi

# Try to use compile_commands.json if available (faster and more accurate)
if [ "$USE_COMPILE_COMMANDS" = true ]; then
    if [ ! -f "build/compile_commands.json" ]; then
        echo "⚠️  compile_commands.json not found. Generating it..."
        cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON >/dev/null 2>&1 || true
        if [ ! -f "build/compile_commands.json" ]; then
            echo "⚠️  Could not generate compile_commands.json, falling back to basic mode"
            USE_COMPILE_COMMANDS=false
        fi
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
echo "$FILES" | xargs -P "$NPROC" -I {} sh -c '
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
for file in $FILES; do
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

echo ""
if [ $HAS_ISSUES -eq 0 ]; then
    echo "✨ All files passed linting! ($CLEAN_FILES files checked)"
    exit 0
else
    echo "❌ Found issues in $FILES_WITH_ISSUES file(s). ($CLEAN_FILES files clean)"
    if [ "$USE_COMPILE_COMMANDS" = false ]; then
        echo "💡 Tip: Run 'pixi run lint-full' for more accurate analysis using compile_commands.json"
    fi
    exit 1
fi


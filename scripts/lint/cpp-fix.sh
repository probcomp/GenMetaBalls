#!/bin/bash
set -e

FILES=$(find genmetaballs/src/cuda tests -type f \( -name '*.cu' -o -name '*.cuh' \) 2>/dev/null || true)

if [ -z "$FILES" ]; then
    echo "✓ No CUDA/C++ files found"
    exit 0
fi

USE_COMPILE_COMMANDS=false
if [ "$1" = "full" ]; then
    USE_COMPILE_COMMANDS=true
    if [ ! -f "build/compile_commands.json" ]; then
        echo "⚠️  compile_commands.json not found. Running 'pixi run compile-commands' first..."
        cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON >/dev/null 2>&1 || true
        if [ ! -f "build/compile_commands.json" ]; then
            echo "❌ Failed to generate compile_commands.json"
            exit 1
        fi
    fi
fi

echo "🔧 Auto-fixing linting issues in CUDA/C++ files..."
FIXED=0
HAS_ISSUES=0

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
        # We can't easily detect what was fixed, so we'll just report the file was processed
        echo "  ✓ Processed: $file"
        FIXED=1
    fi
done <<< "$FILES"

echo ""
echo "✨ Auto-fix complete! ($(echo "$FILES" | wc -l) files processed)"
echo "💡 Note: Review the changes and run 'pixi run lint' to check remaining issues"


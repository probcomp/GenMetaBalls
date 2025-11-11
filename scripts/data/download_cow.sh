#!/bin/bash
# Download cow assets from fuzzy-metaballs repository
# Original source: https://github.com/leonidk/fuzzy-metaballs

set -e  # Exit on error

# Get the project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET_DIR="$PROJECT_ROOT/data/cow"
BASE_URL="https://raw.githubusercontent.com/leonidk/fuzzy-metaballs/refs/heads/main/data"

# Files to download
FILES=(
    "cow.mtl"
    "cow.obj"
    "cow_texture.png"
)

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Download each file
for file in "${FILES[@]}"; do
    SOURCE_URL="$BASE_URL/$file"
    TARGET_FILE="$TARGET_DIR/$file"

    echo "Downloading $file from fuzzy-metaballs repository..."
    wget -O "$TARGET_FILE" "$SOURCE_URL"
done

echo "All files downloaded successfully!"


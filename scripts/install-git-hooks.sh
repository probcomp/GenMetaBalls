#!/bin/bash
# Install git hooks for GenMetaBalls

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"
GIT_HOOKS_SOURCE="$REPO_ROOT/scripts/git-hooks"

if [ ! -d "$HOOKS_DIR" ]; then
    echo "❌ .git/hooks directory not found. Are you in a git repository?"
    exit 1
fi

echo "📦 Installing git hooks..."

# Install pre-commit hook
if [ -f "$GIT_HOOKS_SOURCE/pre-commit" ]; then
    cp "$GIT_HOOKS_SOURCE/pre-commit" "$HOOKS_DIR/pre-commit"
    chmod +x "$HOOKS_DIR/pre-commit"
    echo "✓ Installed pre-commit hook"
else
    echo "⚠️  pre-commit hook not found at $GIT_HOOKS_SOURCE/pre-commit"
    exit 1
fi

# Install pre-push hook
if [ -f "$GIT_HOOKS_SOURCE/pre-push" ]; then
    cp "$GIT_HOOKS_SOURCE/pre-push" "$HOOKS_DIR/pre-push"
    chmod +x "$HOOKS_DIR/pre-push"
    echo "✓ Installed pre-push hook"
else
    echo "⚠️  pre-push hook not found at $GIT_HOOKS_SOURCE/pre-push"
    exit 1
fi

echo ""
echo "✨ Git hooks installed successfully!"
echo "💡 Pre-commit hook: Runs formatting and linting checks before each commit"
echo "💡 Pre-push hook: Runs all tests before allowing push"


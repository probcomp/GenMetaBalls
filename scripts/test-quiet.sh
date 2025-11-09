#!/bin/bash
# Run tests with minimal output on success, full output on failure

set -e

# Run ctest (not using --quiet to get summary)
CTEST_OUTPUT=$(pixi run ctest --test-dir build 2>&1)
CTEST_EXIT=$?

if [ $CTEST_EXIT -ne 0 ]; then
    echo "$CTEST_OUTPUT"
    exit 1
fi

# Extract summary from ctest output (output to stderr so pre-commit shows it)
echo "C++/CUDA tests:" >&2
echo "$CTEST_OUTPUT" | grep -E "(tests passed|Test #.*Passed)" | head -2 | sed 's/^/  /' >&2

# Run pytest with quiet flag
PYTEST_OUTPUT=$(pixi run pytest --quiet 2>&1)
PYTEST_EXIT=$?

if [ $PYTEST_EXIT -ne 0 ]; then
    echo "" >&2
    echo "Python tests:" >&2
    echo "$PYTEST_OUTPUT" >&2
    exit 1
fi

# Extract summary from pytest output (output to stderr so pre-commit shows it)
echo "Python tests:" >&2
echo "$PYTEST_OUTPUT" | grep -E "(passed|failed)" | tail -1 | sed 's/^/  /' >&2

exit 0


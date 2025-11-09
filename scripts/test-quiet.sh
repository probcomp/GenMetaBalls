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

# Extract summary from ctest output
echo "C++/CUDA tests:"
echo "$CTEST_OUTPUT" | grep -E "(tests passed|Test #.*Passed)" | head -2 | sed 's/^/  /'

# Run pytest with quiet flag
PYTEST_OUTPUT=$(pixi run pytest --quiet 2>&1)
PYTEST_EXIT=$?

if [ $PYTEST_EXIT -ne 0 ]; then
    echo ""
    echo "Python tests:"
    echo "$PYTEST_OUTPUT"
    exit 1
fi

# Extract summary from pytest output (quiet mode shows minimal output)
echo "Python tests:"
echo "$PYTEST_OUTPUT" | grep -E "(passed|failed)" | tail -1 | sed 's/^/  /'

exit 0


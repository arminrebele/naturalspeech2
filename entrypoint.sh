#!/bin/bash
set -e

# Create the data directory where paths.py expects it
mkdir -p /workspace/data

# Merge data disks if MERGERFS_DISKS is set
if [ -n "$MERGERFS_DISKS" ]; then
    echo "Merging $MERGERFS_DISKS into /workspace/data..."
    mergerfs -o category.create=mfs,allow_other "$MERGERFS_DISKS" /workspace/data
else
    echo "Warning: MERGERFS_DISKS not set. Skipping mergerfs."
fi

# Build the monotonic_align Cython extension when needed. The .so can't live in the image —
# the bind-mount (./:/workspace) overlays the source tree at runtime; the .so persists on the host.
# Rebuild if EITHER: (1) import fails (missing .so / broken ABI / missing OpenMP), or
# (2) source/build script newer than the .so (same-ABI stale .so = silent perf regression).
MA_DIR=/workspace/naturalspeech2/ops/monotonic_align
NEEDS_BUILD=0

# Import returns the actual loaded .so path → mtime check compares the file Python imports
# (no glob / head picking the wrong .so if multiple ABI tags coexist).
SO_FILE=$(python -c "from naturalspeech2.ops.monotonic_align import core; print(core.__file__)" 2>/dev/null || true)

if [ -z "$SO_FILE" ]; then
    NEEDS_BUILD=1
elif [ "$MA_DIR/core.pyx" -nt "$SO_FILE" ] || [ "$MA_DIR/setup.py" -nt "$SO_FILE" ]; then
    NEEDS_BUILD=1
fi

if [ $NEEDS_BUILD -eq 1 ]; then
    echo "Building monotonic_align Cython extension..."
    (cd "$MA_DIR" && python setup.py build_ext --inplace)
fi

# Run the docker command (default: bash)
exec "$@"

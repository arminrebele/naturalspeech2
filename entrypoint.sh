#!/bin/bash
set -e

# Create the data directory where paths.py expects it
mkdir -p /workspace/data

# Check if MERGERFS_DISKS is provided via environment variable
if [ -n "$MERGERFS_DISKS" ]; then
    echo "Merging $MERGERFS_DISKS into /workspace/data..."
    mergerfs -o category.create=mfs,allow_other "$MERGERFS_DISKS" /workspace/data
else
    echo "Warning: MERGERFS_DISKS not set. Skipping mergerfs."
fi

# Build the monotonic_align Cython extension when needed.
# The .so cannot live in the Docker image because the bind-mount (./:/workspace)
# overlays the source tree at runtime; the .so persists in the host filesystem.
#
# Two complementary checks (rebuild if EITHER triggers):
#   1. Import test — catches missing .so, broken ABI, missing OpenMP linkage.
#   2. Mtime check — catches stale .so (source/build script newer than artifact).
#      The import test alone passes for a same-ABI stale .so, e.g. one built
#      before a `num_threads=b` change in core.pyx — a silent perf regression.
MA_DIR=/workspace/naturalspeech2/ops/monotonic_align
NEEDS_BUILD=0

# Probe with an import that returns the actual loaded .so path. Combining
# both into one Python call means the mtime check is guaranteed to compare
# against the file Python actually imports — no glob, no head -n1 picking
# the wrong .so if multiple ABI tags ever coexist.
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

# Execute the command passed to docker run (defaults to 'bash')
exec "$@"

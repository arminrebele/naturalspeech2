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

# Execute the command passed to docker run (defaults to 'bash')
exec "$@"

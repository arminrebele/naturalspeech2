#!/bin/bash
set -e

# Create the data directory where paths.py expects it
mkdir -p /workspace/data

# Check if both disks are mounted into the container. 
# If they are, run mergerfs to mount them to /workspace/data.
if [ -d "/disk1" ] && [ -d "/disk2" ]; then
    echo "Merging /disk1 and /disk2 into /workspace/data..."
    mergerfs -o category.create=mfs,allow_other /disk1:/disk2 /workspace/data
else
    echo "Warning: /disk1 or /disk2 not found. Skipping mergerfs."
fi

# Execute the command passed to docker run (defaults to 'bash')
exec "$@"

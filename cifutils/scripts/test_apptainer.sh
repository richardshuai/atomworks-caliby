#!/bin/bash
# This script tests a cifutils apptainer container with appropriate bind mounts.
set -e  # Exit on error

echo "Running from $PWD"

# Check if apptainer/singularity is available
APPTAINER_BINARY=$(command -v apptainer || command -v singularity)
if [ -z "$APPTAINER_BINARY" ]; then
    echo "Error: Neither apptainer nor singularity found in PATH"
    exit 1
fi
echo "Using apptainer at: $APPTAINER_BINARY"

# Get the image name from command line argument or use latest
IMAGE_NAME=${1:-cifutils_latest.sif}
if [ ! -f "$IMAGE_NAME" ]; then
    echo "Error: Container image not found: $IMAGE_NAME"
    exit 1
fi

# Initialize BIND_MOUNTS with the base mount
BIND_MOUNTS="$PWD:/cifutils_host"

# Check for standard directories and add them if they exist
echo "Checking for standard directories to mount"
for dir in /projects /net /squash /mnt; do
    if [ -d "$dir" ]; then
        BIND_MOUNTS="$BIND_MOUNTS,$dir:$dir"
        echo "... adding bind mount for: $dir"
    fi
done

# Run `.ipd/setup.sh` in a fault-tolerant way (ignore errors)
echo "Running: .ipd/setup.sh"
echo "----------------------------------------"
.ipd/setup.sh || true
echo "----------------------------------------"

# Add CCD and PDB mirror mounts if environment variables are set
if [ -n "$CCD_MIRROR_PATH" ]; then
    if [ -d "$CCD_MIRROR_PATH" ]; then
        BIND_MOUNTS="$BIND_MOUNTS,$CCD_MIRROR_PATH:$CCD_MIRROR_PATH"
        echo "Adding bind mount for CCD_MIRROR_PATH: $CCD_MIRROR_PATH"
    else
        echo "Warning: CCD_MIRROR_PATH directory does not exist: $CCD_MIRROR_PATH"
    fi
fi

if [ -n "$PDB_MIRROR_PATH" ]; then
    if [ -d "$PDB_MIRROR_PATH" ]; then
        BIND_MOUNTS="$BIND_MOUNTS,$PDB_MIRROR_PATH:$PDB_MIRROR_PATH"
        echo "Adding bind mount for PDB_MIRROR_PATH: $PDB_MIRROR_PATH"
    else
        echo "Warning: PDB_MIRROR_PATH directory does not exist: $PDB_MIRROR_PATH"
    fi
fi

echo "Testing container '$IMAGE_NAME' with bind mounts: '$BIND_MOUNTS'"
echo "Running: $APPTAINER_BINARY test --bind '$BIND_MOUNTS' '$IMAGE_NAME'"
echo "----------------------------------------"
$APPTAINER_BINARY test \
    --bind "$BIND_MOUNTS" \
    "$IMAGE_NAME"
echo "----------------------------------------"

echo
echo "=== Test Complete ==="

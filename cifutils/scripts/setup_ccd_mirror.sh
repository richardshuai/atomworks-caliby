#!/bin/bash

echo "Setting up CCD mirror..."

# Check if path argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <destination_path>"
    exit 1
fi

DEST_PATH="$1"
REMOTE_PATH="rsync://ftp.ebi.ac.uk/pub/databases/msd/pdbechem_v2/ccd/"
ENV_FILE=".env"

echo "Starting CCD mirror setup for target path: $DEST_PATH"

# Create destination directory if it doesn't exist
if [ ! -d "$DEST_PATH" ]; then
    echo "Creating directory: $DEST_PATH"
    mkdir -p "$DEST_PATH"
else
    echo "Directory already exists: $DEST_PATH"
fi

# Perform rsync with file counting
# ... existing code ...

# Perform rsync with file counting
echo "Syncing files from PDBeChem CCD..."
if rsync -rltvz --stats --no-perms --chmod=ug=rwX,o=rX --delete --omit-dir-times \
    --include "*.cif" --exclude "*" \
    "$REMOTE_PATH" "$DEST_PATH"; then
    echo "Sync completed successfully!"

    wget https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz -O "${DEST_PATH}/components.cif.gz"
    # uncompress
    gunzip -f "${DEST_PATH}/components.cif.gz"
else
    echo "Error: Sync failed with exit code $?"
    exit 1
fi

# Create/Update README with sync information
README_PATH="${DEST_PATH}/README"
echo "# CCD Mirror Information" > "$README_PATH"
echo "" >> "$README_PATH"
echo "Last sync: $(date '+%Y-%m-%d %H:%M:%S')" >> "$README_PATH"
echo "Sync script: $(realpath $0)" >> "$README_PATH"
echo "Sync user: $(whoami)" >> "$README_PATH"

# Inform user to set the CCD_MIRROR_PATH environment variable
echo ""
echo "Set 'CCD_MIRROR_PATH=$DEST_PATH' in your .env file"
echo "  ... or run 'export CCD_MIRROR_PATH=$DEST_PATH' in your shell"
echo "  ... or add 'export CCD_MIRROR_PATH=$DEST_PATH' to your .bashrc or .zshrc file to use this path in the code."
echo ""

#!/bin/bash

echo "Setting up RCSB PDB mirror..."

# Check if path argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <destination_path>"
    exit 1
fi

DEST_PATH="$1"
REMOTE_PATH="rsync.wwpdb.org::ftp/data/structures/divided/mmCIF/"
ENV_FILE=".env"


echo "Starting RCSB PDB mirror setup for target path: $DEST_PATH"

# Create destination directory if it doesn't exist
if [ ! -d "$DEST_PATH" ]; then
    echo "Creating directory: $DEST_PATH"
    mkdir -p "$DEST_PATH"
else
    echo "Directory already exists: $DEST_PATH"
fi

# Perform rsync with file counting
echo "Syncing files from RCSB PDB..."
rsync -rltvz --stats --no-perms --chmod=ug=rwX,o=rX --delete --omit-dir-times --port=33444 \
    "$REMOTE_PATH" "$DEST_PATH" > /tmp/rsync_pdb_output.txt 2>&1

echo "Sync complete!"

# Create/Update README with sync information
README_PATH="${DEST_PATH}/README"
echo "# RCSB PDB Mirror Information" > "$README_PATH"
echo "" >> "$README_PATH"
echo "Last sync: $(date '+%Y-%m-%d %H:%M:%S')" >> "$README_PATH"
echo "Sync script: $(realpath $0)" >> "$README_PATH"
echo "Sync user: $(whoami)" >> "$README_PATH"

# Inform user to set the PDB_MIRROR_PATH environment variable
echo ""
echo "Set 'PDB_MIRROR_PATH=$DEST_PATH' in your .env file"
echo "  ... or run 'export PDB_MIRROR_PATH=$DEST_PATH' in your shell"
echo "  ... or add 'export PDB_MIRROR_PATH=$DEST_PATH' to your .bashrc or .zshrc file to use this path in the code."
echo ""

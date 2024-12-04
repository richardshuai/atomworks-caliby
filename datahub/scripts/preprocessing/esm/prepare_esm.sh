#!/bin/bash

# SLURM job configuration
#SBATCH --job-name=Generate_ESM_Embeddings
#SBATCH --output=slurm_log/output_%A_%a.out
#SBATCH --error=slurm_log/err_%A_%a.out
#SBATCH --time=120:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4000:1

# Modify these paths to match your directories
export PDB_CSV_PATH="/projects/ml/RF2_allatom/datasets/pdb/2024_09_10/csv"
export FASTA_INPUT_DIR="./fasta_input_for_esm"
export OUTPUT_BASE_DIR="./RF2_allatom/PDB_ESM_embedding"

# Lock file to indicate extraction completion
EXTRACTION_DONE_FILE="$FASTA_INPUT_DIR/extraction_done"

# Only the first task runs the extraction
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    # Run extraction script
    python extract_sequence_from_csv.py "$PDB_CSV_PATH" "$FASTA_INPUT_DIR"
    
    # Create a "done" file to signal completion
    touch "$EXTRACTION_DONE_FILE"
else
    # Wait until the extraction is finished
    while [ ! -f "$EXTRACTION_DONE_FILE" ]; do
        echo "Waiting for extraction to complete..."
        sleep 10
    done
fi

# Populate the list of files and set up array
FILES=($(ls "$FASTA_INPUT_DIR"))
NUM_FILES=${#FILES[@]}
#SBATCH --array=1-"${NUM_FILES}"

# Set the ESM model and script parameters
ESM_MODEL="esm2_t36_3B_UR50D"
SCRIPT="generate_esm.py"

# Get the current file based on the SLURM array task ID
FILE_NAME=${FILES[$SLURM_ARRAY_TASK_ID-1]}
fasta_filename=$(basename "$FILE_NAME" .fasta)

# Create a dedicated output directory for the current FASTA file's embeddings
output_dir="$OUTPUT_BASE_DIR/$fasta_filename"
mkdir -p "$output_dir"

# Execute the embedding generation with Apptainer
apptainer exec /projects/ml/RF2_allatom/spec_files/datahub_latest.sif \
    python "$SCRIPT" "$ESM_MODEL" "$FASTA_INPUT_DIR/$FILE_NAME" "$output_dir" \
    --repr_layers 36 --include last_layer

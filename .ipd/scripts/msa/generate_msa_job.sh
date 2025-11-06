#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=64g
#SBATCH --gres=gpu:a6000:2
#SBATCH -t 0-06:00:00
#SBATCH -c 6
#SBATCH -C DB
#SBATCH --output=/home/<USER>/slurm_logs/msa_gen_%A_%a.out
#SBATCH --error=/home/<USER>/slurm_logs/msa_gen_%A_%a.err

# Batch MSA Generation Job
# Usage: sbatch --array=1-N .ipd/scripts/msa/generate_msa_job.sh

set -euo pipefail

echo "========================================="
echo "SLURM Job Information"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_JOB_NODELIST}"
echo "Started: $(date)"
echo ""

# ==============================================================================
# Configuration - Read from environment variables with defaults
# ==============================================================================

# Path to apptainer container (relative to job working directory)
APPTAINER_PATH="${APPTAINER_PATH:-.ipd/apptainer/atomworks_dev.sif}"

# Working directory (where manifest.json and chunks/ are located)
WORK_DIR="${WORK_DIR:-./agab_msa_batch}"

# Output directory for MSAs
OUTPUT_DIR="${OUTPUT_DIR:-/projects/msa/lab}"

# Sequence column name
SEQUENCE_COLUMN="${SEQUENCE_COLUMN:-sequence}"

# Existing MSA directories to check (ALWAYS check existing - no option to disable)
EXISTING_MSA_DIRS="${EXISTING_MSA_DIRS:-/projects/msa/lab,/projects/msa/mmseqs_gpu,/projects/msa/hhblits,/projects/msa/mgnify_organized}"

# Number of threads
THREADS="${THREADS:-16}"

# GPU mode (true/false)
USE_GPU="${USE_GPU:-true}"

# MSA tool paths (required by atomworks)
export HHFILTER_PATH="${HHFILTER_PATH:-/net/software/hhsuite/build/bin/hhfilter}"
export MMSEQS2_PATH="${MMSEQS2_PATH:-/net/software/mmseqs-gpu/bin/mmseqs}"

# ColabFold database paths
export COLABFOLD_LOCAL_DB_PATH_GPU="${COLABFOLD_LOCAL_DB_PATH_GPU:-/local/colabfold/gpu}"
export COLABFOLD_LOCAL_DB_PATH_CPU="${COLABFOLD_LOCAL_DB_PATH_CPU:-/local/databases/colabfold/}"
export COLABFOLD_NET_DB_PATH_GPU="${COLABFOLD_NET_DB_PATH_GPU:-/net/databases/colabfold/gpu}"
export COLABFOLD_NET_DB_PATH_CPU="${COLABFOLD_NET_DB_PATH_CPU:-/net/databases/colabfold/}"

# ==============================================================================
# Validate Environment
# ==============================================================================

if [[ ! -f "${APPTAINER_PATH}" ]]; then
    echo "ERROR: Apptainer container not found: ${APPTAINER_PATH}"
    exit 1
fi

if [[ ! -d "${WORK_DIR}" ]]; then
    echo "ERROR: Work directory not found: ${WORK_DIR}"
    exit 1
fi

MANIFEST_PATH="${WORK_DIR}/manifest.json"
if [[ ! -f "${MANIFEST_PATH}" ]]; then
    echo "ERROR: Manifest file not found: ${MANIFEST_PATH}"
    echo "Did you run split_csv_for_msa.py first?"
    exit 1
fi

# ==============================================================================
# Load Chunk Information
# ==============================================================================

# Get chunk filename from manifest
CHUNK_FILE=$(python3 -c "
import json
import sys
with open('${MANIFEST_PATH}') as f:
    manifest = json.load(f)
    chunk_idx = ${SLURM_ARRAY_TASK_ID} - 1  # Convert to 0-indexed
    if chunk_idx >= len(manifest['chunk_files']):
        print(f'ERROR: Invalid array task ID {${SLURM_ARRAY_TASK_ID}}', file=sys.stderr)
        sys.exit(1)
    print(manifest['chunk_files'][chunk_idx])
")

CHUNK_PATH="${WORK_DIR}/chunks/${CHUNK_FILE}"

if [[ ! -f "${CHUNK_PATH}" ]]; then
    echo "ERROR: Chunk file not found: ${CHUNK_PATH}"
    exit 1
fi

echo "========================================="
echo "Configuration"
echo "========================================="
echo "Work Directory: ${WORK_DIR}"
echo "Chunk File: ${CHUNK_FILE}"
echo "Chunk Path: ${CHUNK_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Sequence Column: ${SEQUENCE_COLUMN}"
echo "Existing MSA Dirs: ${EXISTING_MSA_DIRS}"
echo "GPU: ${USE_GPU}"
echo "Threads: ${THREADS}"
echo ""

# ==============================================================================
# Create Output Directory
# ==============================================================================

mkdir -p "${OUTPUT_DIR}"

# ==============================================================================
# Run MSA Generation
# ==============================================================================

echo "========================================="
echo "Running MSA Generation"
echo "========================================="

# Build command based on GPU setting
APPTAINER_FLAGS=""
MSA_FLAGS="--verbose --check-existing --existing-msa-dirs ${EXISTING_MSA_DIRS}"

if [[ "${USE_GPU}" == "true" ]]; then
    APPTAINER_FLAGS="--nv"
    MSA_FLAGS="${MSA_FLAGS} --gpu"
fi

apptainer exec ${APPTAINER_FLAGS} "${APPTAINER_PATH}" \
    atomworks msa generate \
    "${CHUNK_PATH}" \
    "${OUTPUT_DIR}" \
    --sequence-column "${SEQUENCE_COLUMN}" \
    --threads ${THREADS} \
    ${MSA_FLAGS}

# ==============================================================================
# Completion
# ==============================================================================

echo ""
echo "========================================="
echo "Job Completed Successfully"
echo "========================================="
echo "Finished: $(date)"
echo "Chunk: ${CHUNK_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo ""

exit 0

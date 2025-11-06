#!/bin/bash

# Batch MSA Generation Submission Script
# Splits CSV and submits SLURM array jobs for parallel MSA generation

set -euo pipefail

# ==============================================================================
# Default Configuration
# ==============================================================================

CHUNK_SIZE=200
SEQUENCE_COLUMN="sequence"
OUTPUT_DIR="/projects/msa/lab"
CHECK_EXISTING="true"
EXISTING_MSA_DIRS="/projects/msa/lab,/projects/msa/mmseqs_gpu,/projects/msa/hhblits"
USE_GPU="true"
THREADS=16
MEMORY="64G"
TIME="24:00:00"
PARTITION="gpu"
WORK_DIR="${PWD}/msa_batch"

# ==============================================================================
# Usage
# ==============================================================================

usage() {
    cat << EOF
Usage: $0 INPUT_CSV [OPTIONS]

Submit batch MSA generation jobs to SLURM.

Required Arguments:
  INPUT_CSV             Path to CSV file with sequences (one per row)

Options:
  --chunk-size N        Sequences per chunk (default: 200)
  --sequence-column C   Column name with sequences (default: sequence)
  --output-dir DIR      Output directory for MSAs (default: /projects/msa/lab)
  --work-dir DIR        Working directory for chunks (default: ./msa_batch)
  --check-existing      Check for existing MSAs before generation (default: true)
  --no-check-existing   Don't check for existing MSAs
  --existing-msa-dirs   Comma-separated dirs to check (default: /projects/msa/lab,...)
  --gpu                 Use GPU mode (default: true)
  --cpu                 Use CPU mode instead of GPU
  --threads N           Number of threads (default: 16)
  --memory MEM          Memory per job (default: 64G)
  --time TIME           Time limit per job (default: 24:00:00)
  --partition PART      SLURM partition (default: gpu)
  --dry-run             Show what would be done without submitting
  -h, --help            Show this help message

Examples:
  # Basic usage with defaults
  $0 sequences.csv

  # Custom chunk size and output directory
  $0 sequences.csv --chunk-size 500 --output-dir /projects/msa/custom

  # CPU mode with more memory and time
  $0 sequences.csv --cpu --memory 128G --time 48:00:00

  # Don't check for existing MSAs (generate all)
  $0 sequences.csv --no-check-existing

  # Dry run to see what would happen
  $0 sequences.csv --dry-run

EOF
    exit 0
}

# ==============================================================================
# Parse Arguments
# ==============================================================================

if [[ $# -eq 0 ]]; then
    usage
fi

INPUT_CSV=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --sequence-column)
            SEQUENCE_COLUMN="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --check-existing)
            CHECK_EXISTING="true"
            shift
            ;;
        --no-check-existing)
            CHECK_EXISTING="false"
            shift
            ;;
        --existing-msa-dirs)
            EXISTING_MSA_DIRS="$2"
            shift 2
            ;;
        --gpu)
            USE_GPU="true"
            PARTITION="gpu"
            shift
            ;;
        --cpu)
            USE_GPU="false"
            PARTITION="cpu"
            THREADS=32
            MEMORY="128G"
            TIME="48:00:00"
            shift
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --memory)
            MEMORY="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -*)
            echo "ERROR: Unknown option: $1"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
        *)
            if [[ -z "${INPUT_CSV}" ]]; then
                INPUT_CSV="$1"
            else
                echo "ERROR: Unexpected argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# ==============================================================================
# Validate Input
# ==============================================================================

if [[ -z "${INPUT_CSV}" ]]; then
    echo "ERROR: INPUT_CSV is required"
    echo "Run '$0 --help' for usage information"
    exit 1
fi

if [[ ! -f "${INPUT_CSV}" ]]; then
    echo "ERROR: Input CSV not found: ${INPUT_CSV}"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ ! -f "${SCRIPT_DIR}/split_csv_for_msa.py" ]]; then
    echo "ERROR: split_csv_for_msa.py not found in ${SCRIPT_DIR}"
    exit 1
fi

if [[ ! -f "${SCRIPT_DIR}/generate_msa_job.sh" ]]; then
    echo "ERROR: generate_msa_job.sh not found in ${SCRIPT_DIR}"
    exit 1
fi

# ==============================================================================
# Display Configuration
# ==============================================================================

echo "========================================="
echo "Batch MSA Generation Configuration"
echo "========================================="
echo "Input CSV: ${INPUT_CSV}"
echo "Work Directory: ${WORK_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Chunk Size: ${CHUNK_SIZE}"
echo "Sequence Column: ${SEQUENCE_COLUMN}"
echo "Check Existing: ${CHECK_EXISTING}"
if [[ "${CHECK_EXISTING}" == "true" ]]; then
    echo "Existing MSA Dirs: ${EXISTING_MSA_DIRS}"
fi
echo ""
echo "SLURM Configuration:"
echo "  Partition: ${PARTITION}"
echo "  Use GPU: ${USE_GPU}"
echo "  Threads: ${THREADS}"
echo "  Memory: ${MEMORY}"
echo "  Time Limit: ${TIME}"
echo ""

# ==============================================================================
# Step 1: Split CSV into Chunks
# ==============================================================================

echo "========================================="
echo "Step 1: Splitting CSV into Chunks"
echo "========================================="

SPLIT_CMD="python3 ${SCRIPT_DIR}/split_csv_for_msa.py \
    \"${INPUT_CSV}\" \
    \"${WORK_DIR}\" \
    --chunk-size ${CHUNK_SIZE} \
    --sequence-column \"${SEQUENCE_COLUMN}\""

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY RUN] Would run: ${SPLIT_CMD}"
else
    eval "${SPLIT_CMD}"
fi

echo ""

# ==============================================================================
# Step 2: Read Manifest and Prepare SLURM Submission
# ==============================================================================

MANIFEST_PATH="${WORK_DIR}/manifest.json"

if [[ ! -f "${MANIFEST_PATH}" ]] && [[ "${DRY_RUN}" == "false" ]]; then
    echo "ERROR: Manifest file not created: ${MANIFEST_PATH}"
    exit 1
fi

# Get number of chunks
if [[ "${DRY_RUN}" == "true" ]]; then
    NUM_CHUNKS="N"
else
    NUM_CHUNKS=$(python3 -c "import json; print(json.load(open('${MANIFEST_PATH}'))['num_chunks'])")
fi

echo "========================================="
echo "Step 2: Submitting SLURM Array Job"
echo "========================================="
echo "Number of chunks: ${NUM_CHUNKS}"
echo "Array job range: 1-${NUM_CHUNKS}"
echo ""

# Export all configuration variables so they're available to the job script
export WORK_DIR
export OUTPUT_DIR
export SEQUENCE_COLUMN
export EXISTING_MSA_DIRS
export USE_GPU
export THREADS
export APPTAINER_PATH=".ipd/apptainer/atomworks_dev.sif"
export HHFILTER_PATH="/net/software/hhsuite/build/bin/hhfilter"
export MMSEQS2_PATH="/net/software/mmseqs-gpu/bin/mmseqs"
export COLABFOLD_LOCAL_DB_PATH_GPU="/local/colabfold/gpu"
export COLABFOLD_LOCAL_DB_PATH_CPU="/local/databases/colabfold/"
export COLABFOLD_NET_DB_PATH_GPU="/net/databases/colabfold/gpu"
export COLABFOLD_NET_DB_PATH_CPU="/net/databases/colabfold/"

# Build SLURM sbatch command
SBATCH_CMD="sbatch \
    --array=1-${NUM_CHUNKS} \
    --partition=${PARTITION} \
    --cpus-per-task=${THREADS} \
    --mem=${MEMORY} \
    --time=${TIME}"

# Add GPU resources if GPU mode
if [[ "${USE_GPU}" == "true" ]]; then
    SBATCH_CMD="${SBATCH_CMD} --gres=gpu:a6000:1"
fi

# Use --export=ALL to pass all exported environment variables
SBATCH_CMD="${SBATCH_CMD} \
    --export=ALL \
    ${SCRIPT_DIR}/generate_msa_job.sh"

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY RUN] Would submit:"
    echo "${SBATCH_CMD}"
    echo ""
    echo "[DRY RUN] No jobs submitted"
else
    echo "Submitting job..."
    JOB_ID=$(eval "${SBATCH_CMD}" | grep -oP '\d+$')
    echo "Job submitted successfully!"
    echo "Job ID: ${JOB_ID}"
fi

echo ""

# ==============================================================================
# Display Monitoring Commands
# ==============================================================================

echo "========================================="
echo "Monitoring Commands"
echo "========================================="

if [[ "${DRY_RUN}" == "false" ]]; then
    echo "# Check job status"
    echo "squeue -j ${JOB_ID}"
    echo ""
    echo "# Watch job status (updates every 2 seconds)"
    echo "watch -n 2 squeue -j ${JOB_ID}"
    echo ""
    echo "# View specific task output (e.g., task 1)"
    echo "tail -f ~/slurm_logs/msa_gen_${JOB_ID}_1.out"
    echo ""
    echo "# Count completed MSAs"
    echo "ls ${OUTPUT_DIR}/*/*.a3m.gz 2>/dev/null | wc -l"
    echo ""
    echo "# Cancel all tasks"
    echo "scancel ${JOB_ID}"
    echo ""
    echo "# Cancel specific task (e.g., task 5)"
    echo "scancel ${JOB_ID}_5"
else
    echo "# Check job status"
    echo "squeue -j JOB_ID"
    echo ""
    echo "# Watch job status"
    echo "watch -n 2 squeue -j JOB_ID"
    echo ""
    echo "# View task output"
    echo "tail -f ~/slurm_logs/msa_gen_JOB_ID_TASK.out"
    echo ""
    echo "# Count completed MSAs"
    echo "ls ${OUTPUT_DIR}/*/*.a3m.gz 2>/dev/null | wc -l"
fi

echo "========================================="

exit 0

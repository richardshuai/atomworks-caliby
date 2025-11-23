#!/bin/bash
# Submit ASE LMDB validation as SLURM array job (single-threaded, random sampling)
#
# Usage:
#   ./submit_validation.sh [OPTIONS]
#
# Options:
#   --lmdb-path PATH      Path to ASE LMDB (default: /net/tukwila/omol_4m/train_4M)
#   --total-entries N     Total entries in dataset (default: 4000000)
#   --num-jobs N          Number of parallel SLURM jobs (default: 1000)
#   --sample-size N       Entries per job to sample (default: 5000)
#   --output-dir DIR      Output directory (default: ./ase_validation_results)
#   --time TIME           Time limit per job (default: 0:10:00)
#   --mem MEM             Memory per job (default: 4G)
#   --partition PART      SLURM partition (default: cpu)
#   --existing-merged F   Path to existing merged parquet to skip (default: none)
#   --dry-run             Show what would be submitted without submitting

set -e

# Default configuration
LMDB_PATH="/net/tukwila/omol_4m/train_4M"
TOTAL_ENTRIES=4000000
NUM_JOBS=10000
SAMPLE_SIZE=1000
OUTPUT_DIR="./ase_validation_results"
TIME_LIMIT="0:20:00"
MEMORY="4G"
PARTITION="cpu"
EXISTING_MERGED="/home/ncorley/projects/modelhub/latent/lib/atomworks/.ipd/scripts/ase_validation/validation_merged_11_22_2025_14_00.parquet"
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lmdb-path)
            LMDB_PATH="$2"
            shift 2
            ;;
        --total-entries)
            TOTAL_ENTRIES="$2"
            shift 2
            ;;
        --num-jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        --sample-size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --mem)
            MEMORY="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --existing-merged)
            EXISTING_MERGED="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

LAST_JOB_IDX=$((NUM_JOBS - 1))

# Get script directory (absolute path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Convert output dir to absolute path
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$(pwd)/$OUTPUT_DIR"
fi

echo "=========================================="
echo "ASE LMDB Validation - SLURM Array Job"
echo "=========================================="
echo "LMDB Path:       $LMDB_PATH"
echo "Total Entries:   $TOTAL_ENTRIES"
echo "Number of Jobs:  $NUM_JOBS"
echo "Sample Size:     $SAMPLE_SIZE entries per job"
echo "Output Dir:      $OUTPUT_DIR"
echo "Existing Merged: $EXISTING_MERGED"
echo "Time Limit:      $TIME_LIMIT"
echo "Memory:          $MEMORY"
echo "Partition:       $PARTITION"
echo ""
echo "Strategy: Each job samples $SAMPLE_SIZE random indices,"
echo "          skipping any already processed by other jobs."
echo "          No multiprocessing - single-threaded per job."
echo "=========================================="

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[DRY RUN] Would submit the following job:"
    echo ""
fi

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# Create the job script
JOB_SCRIPT="$OUTPUT_DIR/validation_job.sh"
cat > "$JOB_SCRIPT" << 'JOBSCRIPT'
#!/bin/bash
#SBATCH --job-name=validate_ase
#SBATCH --output=OUTPUT_DIR_PLACEHOLDER/logs/validate_%a.out
#SBATCH --error=OUTPUT_DIR_PLACEHOLDER/logs/validate_%a.err
#SBATCH --time=TIME_PLACEHOLDER
#SBATCH --cpus-per-task=1
#SBATCH --mem=MEM_PLACEHOLDER
#SBATCH --partition=PARTITION_PLACEHOLDER
#SBATCH --array=0-LAST_JOB_PLACEHOLDER

# Configuration (set by submit script)
LMDB_PATH="LMDB_PLACEHOLDER"
OUTPUT_DIR="OUTPUT_DIR_PLACEHOLDER"
TOTAL_ENTRIES=TOTAL_PLACEHOLDER
SAMPLE_SIZE=SAMPLE_PLACEHOLDER
SCRIPT_DIR="SCRIPT_DIR_PLACEHOLDER"
EXISTING_MERGED="EXISTING_MERGED_PLACEHOLDER"

echo "=========================================="
echo "Job ID: $SLURM_ARRAY_TASK_ID"
echo "Sampling $SAMPLE_SIZE random indices from $TOTAL_ENTRIES total"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="
echo

# Activate environment - atomworks is in the parent repo
cd "$SCRIPT_DIR/../../../../"
source .venv/bin/activate

# Run validation (single-threaded, random sampling)
python "$SCRIPT_DIR/validate_ase.py" \
    --output-dir "$OUTPUT_DIR" \
    --chunk-id $SLURM_ARRAY_TASK_ID \
    --sample-size $SAMPLE_SIZE \
    --lmdb-path "$LMDB_PATH" \
    --total-entries $TOTAL_ENTRIES \
    --checkpoint-interval 10 \
    --existing-merged "$EXISTING_MERGED"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Job completed successfully"
else
    echo "WARNING: Job exited with code $EXIT_CODE (partial results may be saved)"
fi

exit $EXIT_CODE
JOBSCRIPT

# Replace placeholders
sed -i "s|OUTPUT_DIR_PLACEHOLDER|$OUTPUT_DIR|g" "$JOB_SCRIPT"
sed -i "s|TIME_PLACEHOLDER|$TIME_LIMIT|g" "$JOB_SCRIPT"
sed -i "s|MEM_PLACEHOLDER|$MEMORY|g" "$JOB_SCRIPT"
sed -i "s|PARTITION_PLACEHOLDER|$PARTITION|g" "$JOB_SCRIPT"
sed -i "s|LAST_JOB_PLACEHOLDER|$LAST_JOB_IDX|g" "$JOB_SCRIPT"
sed -i "s|LMDB_PLACEHOLDER|$LMDB_PATH|g" "$JOB_SCRIPT"
sed -i "s|TOTAL_PLACEHOLDER|$TOTAL_ENTRIES|g" "$JOB_SCRIPT"
sed -i "s|SAMPLE_PLACEHOLDER|$SAMPLE_SIZE|g" "$JOB_SCRIPT"
sed -i "s|SCRIPT_DIR_PLACEHOLDER|$SCRIPT_DIR|g" "$JOB_SCRIPT"
sed -i "s|EXISTING_MERGED_PLACEHOLDER|$EXISTING_MERGED|g" "$JOB_SCRIPT"

if [ "$DRY_RUN" = true ]; then
    echo "Job script saved to: $JOB_SCRIPT"
    echo ""
    echo "To submit manually:"
    echo "  sbatch $JOB_SCRIPT"
    exit 0
fi

# Submit the job
echo ""
echo "Submitting SLURM array job..."
JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
echo "Submitted job: $JOB_ID"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER -n validate_ase"
echo "  ./check_status.sh --output-dir $OUTPUT_DIR"
echo ""
echo "After completion, merge results with:"
echo "  python merge_results.py --input-dir $OUTPUT_DIR"

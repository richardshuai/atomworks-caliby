#!/bin/bash
#SBATCH --job-name=validate_ase
#SBATCH --output=/home/ncorley/projects/modelhub/latent/lib/atomworks/.ipd/scripts/ase_validation/./ase_validation_results/logs/validate_%a.out
#SBATCH --error=/home/ncorley/projects/modelhub/latent/lib/atomworks/.ipd/scripts/ase_validation/./ase_validation_results/logs/validate_%a.err
#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=cpu
#SBATCH --array=0-9999

# Configuration (set by submit script)
LMDB_PATH="/net/tukwila/omol_4m/train_4M"
OUTPUT_DIR="/home/ncorley/projects/modelhub/latent/lib/atomworks/.ipd/scripts/ase_validation/./ase_validation_results"
TOTAL_ENTRIES=4000000
SAMPLE_SIZE=1000
SCRIPT_DIR="/home/ncorley/projects/modelhub/latent/lib/atomworks/.ipd/scripts/ase_validation"
EXISTING_MERGED="/home/ncorley/projects/modelhub/latent/lib/atomworks/.ipd/scripts/ase_validation/validation_merged_11_22_2025_14_00.parquet"

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

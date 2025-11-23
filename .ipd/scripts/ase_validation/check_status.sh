#!/bin/bash
# Check status of ASE validation SLURM array jobs
#
# Usage:
#   ./check_status.sh [--output-dir DIR]

OUTPUT_DIR="./ase_validation_results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "ASE Validation Job Status"
echo "=========================================="
echo

# Check SLURM job status
echo "SLURM Jobs:"
squeue -u $USER -n validate_ase -o "%.18i %.9P %.30j %.8T %.10M %.6D %R" 2>/dev/null || echo "  No jobs found in queue"
echo

# Check output directory
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Output directory does not exist yet: $OUTPUT_DIR"
    exit 0
fi

echo "Output Files:"
CHUNK_FILES=$(find "$OUTPUT_DIR" -name "chunk_*.parquet" 2>/dev/null | wc -l)
ATTEMPTED_FILES=$(find "$OUTPUT_DIR" -name "*.attempted" 2>/dev/null | wc -l)
echo "  Chunk files completed: $CHUNK_FILES"
echo "  Jobs in progress (attempted files): $ATTEMPTED_FILES"

if [ $CHUNK_FILES -gt 0 ]; then
    echo
    echo "Entries Processed:"
    python3 -c "
import sys
from pathlib import Path
import pandas as pd

output_dir = Path('$OUTPUT_DIR')
chunk_files = sorted(output_dir.glob('chunk_*.parquet'))

if not chunk_files:
    print('  No chunk files found')
    sys.exit(0)

total_entries = 0
rdkit_failed = 0
pb_failed = 0

for chunk_file in chunk_files:
    try:
        df = pd.read_parquet(chunk_file)
        total_entries += len(df)
        rdkit_failed += df['rdkit_conversion_failed'].sum()
        pb_failed += df['pb_bust_failed'].sum()
    except Exception as e:
        print(f'  ERROR reading {chunk_file.name}: {e}')

print(f'  Total entries processed: {total_entries:,}')
print()
print('Validation Results:')
rdkit_ok = total_entries - rdkit_failed
pb_ok = total_entries - pb_failed
print(f'  RDKit conversion success: {rdkit_ok:,}/{total_entries:,} ({rdkit_ok/total_entries*100:.1f}%)')
print(f'  PoseBusters success: {pb_ok:,}/{total_entries:,} ({pb_ok/total_entries*100:.1f}%)')

# Check for PoseBusters all_passed if available
try:
    sample_df = pd.read_parquet(chunk_files[0])
    if 'pb_mol_pred_loaded' in sample_df.columns:
        # Aggregate pb_ columns
        all_dfs = [pd.read_parquet(f) for f in chunk_files]
        merged = pd.concat(all_dfs, ignore_index=True)
        pb_cols = [c for c in merged.columns if c.startswith('pb_') and c not in ['pb_bust_failed']]
        if pb_cols:
            # All PB checks passed
            all_passed = merged[pb_cols].all(axis=1).sum()
            print(f'  All PB checks passed: {all_passed:,}/{total_entries:,} ({all_passed/total_entries*100:.1f}%)')
except Exception:
    pass
" 2>/dev/null || echo "  Could not read parquet files"
fi

echo
echo "=========================================="

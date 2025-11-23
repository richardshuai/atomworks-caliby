# ASE LMDB Validation

Validate ASE LMDB datasets (e.g., OMol25) at scale using SLURM array jobs. Tests RDKit conversion and PoseBusters chemical validity checks.

## Quick Start

```bash
# Submit parallel jobs to validate ~4M entries
./.ipd/scripts/ase_validation/submit_validation.sh

# Check progress
./.ipd/scripts/ase_validation/check_status.sh

# After completion, merge results
python ./.ipd/scripts/ase_validation/merge_results.py
```

### Incremental Validation (Resume from Previous Run)

To continue validation from a previous merged result file:

```bash
# Submit jobs, skipping entries already in an existing merged file
./submit_validation.sh --existing-merged validation_merged_previous.parquet

# After completion, merge new chunks with the existing merged file
python merge_results.py \
    --input-dir ./ase_validation_results \
    --output validation_merged_combined.parquet \
    --existing-merged validation_merged_previous.parquet
```

## Usage

### Submit Validation Jobs

```bash
./submit_validation.sh [OPTIONS]
```

**Options:**
- `--lmdb-path PATH` - Path to ASE LMDB database (default: `/net/tukwila/omol_4m/train_4M`)
- `--total-entries N` - Total entries to validate (default: 4000000)
- `--num-jobs N` - Number of parallel SLURM jobs (default: 10000)
- `--sample-size N` - Entries per job to sample (default: 1000)
- `--output-dir DIR` - Directory for output files (default: `./ase_validation_results`)
- `--time TIME` - Time limit per job (default: `0:20:00`)
- `--mem MEM` - Memory per job (default: `8G`)
- `--partition PART` - SLURM partition (default: `cpu`)
- `--existing-merged FILE` - Path to existing merged parquet to skip already-processed indices
- `--dry-run` - Show what would be submitted without submitting

**Examples:**

```bash
# Default: 100 jobs for 4M entries
./submit_validation.sh

# Custom configuration
./submit_validation.sh --num-jobs 200 --total-entries 10000000

# Dry run to see configuration
./submit_validation.sh --dry-run
```

### Check Status

```bash
./check_status.sh [--output-dir DIR]
```

Shows:
- Active SLURM jobs
- Completed chunk files
- Entry counts and validation statistics

### Merge Results

```bash
python merge_results.py --input-dir ./ase_validation_results --output merged.parquet

# Incremental merge with existing merged file:
python merge_results.py \
    --input-dir ./ase_validation_results \
    --output merged_combined.parquet \
    --existing-merged previous_merged.parquet
```

**Options:**
- `--input-dir DIR` - Directory containing chunk_*.parquet files (default: `./ase_validation_results`)
- `--output FILE` - Output path for merged parquet file
- `--existing-merged FILE` - Optional existing merged parquet to include in merge

Combines all chunk files (and optionally an existing merged file) into a single parquet with:
- Deduplication (keeps first occurrence)
- Sorting by LMDB index
- Detailed per-check statistics

## Output Format

### Chunk Files

Each chunk file (`chunk_XXXX.parquet`) contains:

| Column | Type | Description |
|--------|------|-------------|
| `lmdb_idx` | int | Original LMDB index |
| `num_atoms` | int | Number of atoms |
| `source` | str | Source identifier |
| `charge` | int | Molecular charge |
| `energy` | float | DFT energy |
| `rdkit_conversion_failed` | bool | RDKit conversion failed |
| `pb_bust_failed` | bool | PoseBusters validation failed |
| `pb_*` | bool | Individual PoseBusters check results |

### PoseBusters Checks

When validation succeeds, includes columns for each PoseBusters check:
- `pb_mol_pred_loaded` - Molecule loaded successfully
- `pb_sanitization` - RDKit sanitization passed
- `pb_all_atoms_connected` - No disconnected fragments
- `pb_bond_lengths` - Bond lengths within expected ranges
- `pb_bond_angles` - Bond angles within expected ranges
- `pb_internal_steric_clash` - No internal clashes
- ... and more

## Crash Recovery

The validation script includes automatic crash recovery:

1. **Checkpoint files** - Results saved every 500 entries
2. **Attempted tracking** - If a specific entry crashes, it's skipped on restart
3. **Auto-retry** - Jobs retry up to 10 times on crash

If a job crashes repeatedly on a specific entry, that entry will be marked in `.attempted` files and skipped.

## Scripts

| Script | Description |
|--------|-------------|
| `submit_validation.sh` | Main entry point - submits SLURM array job |
| `validate_ase.py` | Core validation logic (called by SLURM jobs) |
| `check_status.sh` | Monitor job progress and results |
| `merge_results.py` | Combine chunk files into single output |

## Requirements

- Python 3.11+
- atomworks (with RDKit support)
- posebusters
- pandas
- fire
- tqdm

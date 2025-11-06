# Batch MSA Generation

Generate MSAs at scale using SLURM array jobs. Automatically splits CSV files into chunks and processes them in parallel.

## Quick Start

```bash
# Basic usage (GPU mode, 200 sequences per chunk)
./.ipd/scripts/msa/submit_msa_generation.sh sequences.csv

# Custom chunk size
./.ipd/scripts/msa/submit_msa_generation.sh sequences.csv --chunk-size 500

# Dry run (see what would happen)
./.ipd/scripts/msa/submit_msa_generation.sh sequences.csv --dry-run
```

## Usage

### Input Format

CSV file with a `sequence` column (or specify `--sequence-column NAME`):

```csv
sequence
MKKKEVESGGGLVQPGGSLRLSCAASGFTISDYWIHWVRQAPG...
MSYIWRQLTGSKVKDVTFSCASQVTQTYAMSWVRQPPGKGLE...
```

### Submit Jobs

```bash
./submit_msa_generation.sh sequences.csv [OPTIONS]
```

**Common Options:**
- `--chunk-size N` - Sequences per chunk (default: 200)
- `--output-dir DIR` - Output directory (default: `/projects/msa/lab`)
- `--no-check-existing` - Don't check for existing MSAs
- `--dry-run` - Show what would happen without submitting

**Full Options:**
```
--chunk-size N           Sequences per chunk (default: 200)
--sequence-column NAME   Column with sequences (default: sequence)
--output-dir DIR         Output directory (default: /projects/msa/lab)
--work-dir DIR           Working directory for chunks (default: ./msa_batch)
--check-existing         Check for existing MSAs (default: enabled)
--no-check-existing      Don't check for existing MSAs
--gpu                    Use GPU mode (default: enabled)
--cpu                    Use CPU mode instead
--threads N              Number of threads (default: 16 GPU, 32 CPU)
--memory MEM             Memory per job (default: 64G GPU, 128G CPU)
--time TIME              Time limit (default: 24:00:00 GPU, 48:00:00 CPU)
--partition PART         SLURM partition (default: gpu)
--dry-run                Show what would be done without submitting
```

## Output

MSAs are written to `/projects/msa/lab` (or `--output-dir`) with sharded structure:

```
/projects/msa/lab/
├── 00/00a1b2c3d4e.a3m.gz
├── 01/01f5e4d3c2b.a3m.gz
└── ...
```

- **Sharding:** 256 subdirs (`00-ff`) based on sequence hash
- **Naming:** First 11 chars of SHA-256 hash
- **Format:** Compressed a3m (`.a3m.gz`)

## Scripts

### `submit_msa_generation.sh`
Main entry point. Splits CSV and submits SLURM array job.

### `generate_msa_job.sh`
SLURM job script that processes one chunk. Reads configuration from environment variables.

### `split_csv_for_msa.py`
Splits CSV into chunks. Creates `chunks/` directory and `manifest.json`.

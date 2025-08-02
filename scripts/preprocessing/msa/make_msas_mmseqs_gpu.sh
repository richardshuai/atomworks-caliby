#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=124g
#SBATCH --gres=gpu:a6000:2
#SBATCH -t 0-06:00:00
#SBATCH -c 6
#SBATCH -C DB
#SBATCH -o /home/<USER>/slurm_logs/slurm_%A_%a.log  --open-mode=append
#SBATCH -e /home/<USER>/slurm_logs/slurm_%A_%a.err  --open-mode=append


# This works if you're running from this directory. Otherwise you could use something like this: /home/<USER>/projects/datahub/.ipd/datahub.sif
APPTAINER_PATH="${PWD}/../../../.ipd/datahub.sif"

echo "Starting task ${SLURM_ARRAY_TASK_ID} on ${SLURM_JOB_NODELIST}"

# Add datahub to the PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/home/<USER>/projects/datahub/src"

# Default values for the arguments
out_dir="/net/scratch/<USER>/msa_output/"
input_file="/home/<USER>/msas/msas_to_process.csv"
seq_col="q_pn_unit_processed_entity_non_canonical_sequence"
chain_type_col="q_pn_unit_type"  # If this isn't in your dataframe it is ignored

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --out_dir) out_dir="$2"; shift ;;
        --input_file) input_file="$2"; shift ;;
        --seq_col) seq_col="$2"; shift ;;
        --chain_type_col) chain_type_col="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Make the directory to store the files, if it doesn't exist
echo "Making directory: ${out_dir}"
mkdir -p "${out_dir}"

# Print the parameters
echo "Saving files to: ${out_dir}"

# Get the directory of the current script
echo "Script directory: ${PWD}"

# Run the Python script
apptainer exec --nv "${APPTAINER_PATH}" \
python "${PWD}/make_msa_mmseqs_colabfold.py" \
    --input_file "${input_file}" \
    --output_dir "${out_dir}" \
    --gpu \
    --gpu_server \
    --sequence_col "${seq_col}" \
    --chain_type_col "${chain_type_col}"

# Example usage:
# sbatch ./make_msas_mmseqs_gpu.sh --out_dir /path/to/output/directory --input_file /path/to/input/csv_file.csv --seq_col "sequence" --chain_type_col ""

#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem-per-cpu=64g
#SBATCH -a 1-500
#SBATCH -t 0-12:00:00
#SBATCH -c 1
#SBATCH -o slurm_logs/slurm_%A_%a.log
#SBATCH -e slurm_logs/slurm_%A_%a.err

# Calculate task ID and number of tasks
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))
NUM_TASKS=$SLURM_ARRAY_TASK_COUNT

# Set PYTHONPATH to the base project directory (will need to modify if the location of this script moves)
export PYTHONPATH="${PWD}:${PWD}/../../../:$PYTHONPATH"
echo "PYTHONPATH: ${PYTHONPATH}"

# Default values for the arguments
out_dir="/projects/ml/RF2_allatom/data_preprocessing/${USER}/${SLURM_JOB_ID}"
base_cif_dir="/databases/rcsb/cif"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --out_dir) out_dir="$2"; shift ;;
        --base_cif_dir) base_cif_dir="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Make the directory to store the files, if it doesn't exist
mkdir -p "${out_dir}"

# Create a README file if it doesn't exist, and add timestamp, PDB directory path, and base_cif_dir
readme_file="${out_dir}/README"
if [ ! -f "${readme_file}" ]; then
    echo "Creating README file at ${readme_file}"
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    {
        echo "Timestamp: ${timestamp}"
        echo "PDB Directory Path: ${out_dir}"
        echo "CIF Directory Path: ${base_cif_dir}"
    } >> "${readme_file}"
fi

# Print the parameters
echo "Saving files to: ${out_dir}"

### Run the Python script
command="apptainer exec /projects/ml/RF2_allatom/spec_files/datahub_latest.sif python process_pdbs.py --task_id ${TASK_ID} --num_tasks ${NUM_TASKS} --out_dir '${out_dir}' --pdb_selection 'all' --base_cif_dir '${base_cif_dir}'"
echo -e "command\t$command"
eval $command

# Example usage:
# sbatch data/scripts/process_pdbs.sh --out_dir /path/to/output/directory --base_cif_dir /path/to/cif/directory

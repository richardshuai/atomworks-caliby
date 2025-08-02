#!/bin/bash
#SBATCH -p cpu
#SBATCH --job-name=count_af3_tokens
#SBATCH -o slurm_logs/slurm_%A_%a.log
#SBATCH -e slurm_logs/slurm_%A_%a.err
#SBATCH --array=0-50
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16g
#SBATCH --time=04:00:00

# Calculate task ID and number of tasks
TASK_ID=$((SLURM_ARRAY_TASK_ID))
NUM_TASKS=$SLURM_ARRAY_TASK_COUNT

# Set PYTHONPATH to the base project directory (will need to modify if the location of this script moves)
export PYTHONPATH="${PWD}/../../../:$PYTHONPATH"
echo "PYTHONPATH: ${PYTHONPATH}"

# Run the Python script
command="apptainer exec /projects/ml/RF2_allatom/spec_files/datahub_latest.sif python count_af3_tokens.py --pn_units_df_path /projects/ml/RF2_allatom/datasets/pdb/2024_09_10/pn_units_df.parquet  --num_workers ${SLURM_CPUS_PER_TASK} --task_id ${TASK_ID} --num_tasks ${NUM_TASKS}"
echo -e "command\t$command"
eval $command

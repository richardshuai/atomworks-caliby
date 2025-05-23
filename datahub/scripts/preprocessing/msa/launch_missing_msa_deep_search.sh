#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=196g
#SBATCH -t 0-08:00:00
#SBATCH -c 12
#SBATCH -o /home/<USER>/slurm_logs/slurm_%A_%a.log  --open-mode=append
#SBATCH -e /home/<USER>/slurm_logs/slurm_%A_%a.err  --open-mode=append


# This works if you're running from this directory. Otherwise you could use something like this: /home/<USER>/projects/datahub/.ipd/datahub.sif
APPTAINER_PATH="${PWD}/../../../.ipd/datahub.sif"

echo "Starting task ${SLURM_ARRAY_TASK_ID} on ${SLURM_JOB_NODELIST}"

# Add datahub to the PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/home/<USER>/projects/datahub/src:/home/<USER>/projects/cifutils/src"

df_path="/projects/ml/datahub/dfs/pdb/2024_12_01/interfaces_df.parquet"
msa_dirs="/projects/msa/rf2aa_af3/rf2aa_paper_model_protein_msas,/projects/msa/rf2aa_af3/missing_msas_through_2024_08_12,/net/scratch/mkazman/msa/validate_no_leak_taxid,/net/scratch/mkazman/msa/missing_antibody_msas,/net/scratch/mkazman/msa/post_training_cutoff_msas/processed_nested,/net/scratch/mkazman/msa/post_training_cutoff_msas/extra_seqs_processed_nested,/projects/msa/nvidia_renamed_with_seq_hash/maxseq_10k"
output_path=""  # fill this in
num_workers=12


# Run the Python script
apptainer exec "${APPTAINER_PATH}" \
python "${PWD}/find_missing_msa_protein_sequences.py" \
    --df_path "${df_path}" \
    --msa_dirs "${msa_dirs}" \
    --output_path "${output_path}" \
    --deep_search \
    --num_workers "${num_workers}"

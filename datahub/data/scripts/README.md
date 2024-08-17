# Pipeline Overview

This README provides an overview of the pipeline used for processing PDB files, generating dataframes, clustering sequences, annotating clusters, and generating interfaces dataframes.

## Pipeline Flow

1. **Generate a CSV file for each PDB example**
   - **Script:** `process_pdbs.py` (or `process_pdbs.sh`)
   - **Environment:** Run on `DIGS`
   - **Resources:** ~300 nodes
   - **Duration:** Approximately ~6 hours, depending on how many nodes are allocated
   - **Tip:** Launch ~100 nodes with `64g` with `SLURM` priority (`cpu`), and ~200 nodes with `32g` on backfill (`cpu-bf`)
   - **Example:** `sbatch process_pdbs.sh --out_dir /projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_05`

2. **Concatenate the CSV files into the PN Units DataFrame**
   - **Script:** `generate_pn_units_df.py`
   - **Environment:** Run on `jojo` with `tmux`
   - **Duration:** Approximately ~1 hour
   - **Example:** `python generate_pn_units_df.py --input_dir /projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_05/csv --output_path /projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_05/pn_units_df`

3. **Cluster Sequences**
   - **Script:** `cluster_sequences.py`
   - **Description:** Clusters the sequences based on the generated PN units dataframe.
   - **Environment:** Run on `jojo` with `tmux`
   - **Duration:** Approximately ~10 minutes
   - **Example:** `python cluster_sequences.py --pn_units_df /projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_05/pn_units_df.parquet --output_path /projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_05/pn_units_df_clustered.parquet`

4. **Annotate the Clusters**
   - **Script:** `annotate_clusters.py`
   - **Description:** Annotates the clusters with relevant information.
   - **Environment:** Run on `jojo` with `tmux`
   - **Duration:** Approximately ~5 minutes
   - **Example:** `python annotate_clusters.py --pn_units_df /projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_05/pn_units_df_clustered.parquet --replace_df`

5. **Generate Interfaces DataFrame**
   - **Script:** `generate_interfaces_df.py`
   - **Description:** Generates the interfaces dataframe using the annotated clusters.
   - **Environment:** Run on `jojo` with `tmux`
   - **Duration:** Approximately ~10 minutes
   - **Example:** `python generate_interfaces_df.py --input_path /projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_05/pn_units_df_clustered.parquet --output_path /projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_05/interfaces_df.parquet`

## Contributing
A helpful contribution would be a unified script that launches timed jobs that perform the above steps sequentially, notifying the user when the process is complete.

The initial version of this codebase was written by [Nate Corley](mailto:ncorley@uw.edu) and [Simon Mathis](mailto:simon.mathis@gmail.com) in the summer of 2024, in preparation for training an AF3-like structure prediction model. We hope that others will contribute to improve, and extend, our codebase.
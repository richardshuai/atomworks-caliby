# Pipeline Overview

This README provides an overview of the pipeline used for processing structure files, generating pn_units dataframes, clustering sequences, and generating interfaces dataframes.

This pipeline can be used for structures from the PDB or for distillation structures, as long as the `from_pdb` argument for `get_csvs_from_structures` is modified accordingly.

**Note:** This functionality can be accessed either by importing python functions (see Example A) or by running the associated files as scripts (see Example B).

## Pipeline Flow

### Example A: Small dataset, importing functions

1. **Generate a CSV file for each structure**

   ```python
   from pathlib import Path
   from scripts.preprocessing.pdb import get_csvs_from_structures

   # Declare paths and structure file extension
   INPUT_DIR = Path("/path/to/input/structures")
   OUTPUT_DIR = Path("/path/to/output/dataframes")
   STRUCTURE_FILE_EXTENSION=".pdb" 

   # Get csv files for each structure
   get_csvs_from_structures.run_pipeline(base_dir=INPUT_DIR, out_dir = OUTPUT_DIR, from_pdb = False, file_extension=STRUCTURE_FILE_EXTENSION)
   ```

2. **Concatenate the CSV files into the PN Units DataFrame**

   ```python
   from scripts.preprocessing.pdb.generate_pn_units_df import generate_pn_units_df

   # Combine the csv files into a single pn_units dataframe
   generate_pn_units_df(OUTPUT_DIR / "csv", OUTPUT_DIR / "pn_units_df.parquet", num_workers=1, dataset_name = 'my_example_dataset')
   ```

3. **Cluster Sequences (optional)**
   - **Option A:** Cluster like AF3

   ```python
   from scripts.preprocessing.clustering.cluster_sequences import cluster_and_annotate_clusters

   cluster_and_annotate_clusters(pn_units_df = OUTPUT_DIR / "pn_units_df.parquet", output_path = OUTPUT_DIR / "clustered_pn_units_df.parquet")
   ```

   - **Option B**: Custom clustering workflow (example: clustering on concatenated CDRs)
      - Subset the pn_units_df as appropriate if performing distinct clustering operations
      - For each such subset, cluster using `datahub.preprocessing.utils.clustering.cluster_all_sequences`
      - Recombine the clustered subsets into a single dataframe, and add a column named "cluster" containing each pn_unit's cluster information

4. **Generate Interfaces DataFrame**

   ```python
   from scripts.preprocessing.pdb.generate_interfaces_df import generate_and_save_interfaces_df

   # Generate the corresponding interfaces dataframe
   generate_and_save_interfaces_df(OUTPUT_DIR / "clustered_pn_units_df.parquet", OUTPUT_DIR / "interfaces_df.parquet")
   ```

### Example B: Large dataset, running scripts

**Note:** Timing/resource estimates are for running on the entire PDB, containing ~225,000 structures as of this writing.

1. **Generate a CSV file for each structure**
   - **Script:** `get_csvs_from_structures.py` (or `get_csvs_from_structures.sh`)
   - **Environment:** Run on `DIGS`
   - **Resources:** ~300 nodes
   - **Duration:** Approximately ~6 hours, depending on how many nodes are allocated
   - **Tip:** Launch ~100 nodes with `64g` with `SLURM` priority (`cpu`), and ~200 nodes with `32g` on backfill (`cpu-bf`)
   - **Example:** `sbatch get_csvs_from_structures.sh --from_pdb True --file_extension .cif.gz --base_dir /databases/rcsb/cif --out_dir /projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_05`

2. **Concatenate the CSV files into the PN Units DataFrame**
   - **Script:** `generate_pn_units_df.py`
   - **Environment:** Run on `jojo` with `tmux`
   - **Duration:** Approximately ~1 hour
   - **Example:** `python generate_pn_units_df.py --input_dir /projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_05/csv --output_path /projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_05/pn_units_df.parquet --dataset_name my_example_dataset`

3. **Cluster Sequences (optional)**
   - **Option A:** Cluster like AF3
      - **Script:** `cluster_sequences.py`
      - **Description:** Clusters the sequences based on chain type (as in AF3), and creates a general `cluster` column.
      - **Environment:** Run on `jojo` with `tmux`
      - **Duration:** Approximately ~15 minutes
      - **Example:** `python cluster_sequences.py --pn_units_df /projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_05/pn_units_df.parquet --output_path /projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_05/pn_units_df_clustered.parquet`
   - **Option B**: Custom clustering workflow (example: clustering on concatenated CDRs)
      - Subset the pn_units_df as appropriate if performing distinct clustering operations
      - For each such subset, cluster using `datahub.preprocessing.utils.clustering.cluster_all_sequences`
      - Recombine the clustered subsets into a single dataframe, and add a column named "cluster" containing each pn_unit's cluster information

4. **Generate Interfaces DataFrame**
   - **Script:** `generate_interfaces_df.py`
   - **Description:** Generates the interfaces dataframe, using the annotated clusters if present.
   - **Environment:** Run on `jojo` with `tmux`
   - **Duration:** Approximately ~10 minutes
   - **Example:** `python generate_interfaces_df.py --input_path /projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_05/pn_units_df_clustered.parquet --output_path /projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_05/interfaces_df.parquet`

## MSAs

Although not described in the above pipeline, protein (but not RNA) MSAs can be computed with `hhsuite` via the scripts found in the `msa` subfolder.

To train with new MSA's (e.g., for distillation):
   1. Create the MSA's with your tool of choice, saved in a `a3m` format on the `digs`
   2. Name the MSA files with the sequence hash of the polymer (see [here]([url](https://github.com/baker-laboratory/datahub/blob/main/datahub/utils/misc.py#L81)) for the hashing function, and [here]([url](https://github.com/baker-laboratory/datahub/blob/628c2011c15b6e48ade8413a50c2b4f5cbd410cc/scripts/preprocessing/msa/rename_msa_files_with_seq_hash.py#L4)) for an example script that renames all `a3m` files in a directory with their appropriate sequence hash). This naming schema is how we lookup MSA files. 
   3. If there are ~1K MSA files, shard your MSA directory with the letters of the sequence hash. For an example, see the directory structure in `/projects/msa/rf2aa_af3/rf2aa_paper_model_protein_msas`
   4. Move the MSA files to `/projects/msa`, or elsewhere on the `digs` that is generally accessible and has sufficent storage space (NOTE: If generating many thousands of MSA's, we should store the MSA's as a `squashfs`, like was done for the Facebook distillation dataset)
   5. Add the path of the MSA directory, along with the file extension and the sharding depth, to the `hydra` config for the model you wish to train. For example, in the current config, this looks like:
```yaml
protein_msa_dirs:
    - {"dir": "/projects/msa/rf2aa_af3/rf2aa_paper_model_protein_msas", "extension": ".a3m.gz", "directory_depth": 2}
    - {"dir": "/projects/msa/rf2aa_af3/missing_msas_through_2024_08_12", "extension": ".msa0.a3m.gz", "directory_depth": 2}
```
These paths will be searched in order to find an MSA matching the sequence has of each polymer in an example.

We welcome additional scripts and documentation within this area, especially with respect to computing MSAs for RNA and with `MMSeqs2` vs. `hhblits`.

## Contributing

A helpful contribution would be a unified script that launches timed jobs that perform the above steps sequentially, notifying the user when the process is complete.

The initial version of this codebase was written by [Nate Corley](mailto:ncorley@uw.edu) and [Simon Mathis](mailto:simon.mathis@gmail.com) in the summer of 2024, in preparation for training an AF3-like structure prediction model. Additional preprocessing functionality was added by [Rafi Brent](mailto:rib7@uw.edu) in the fall of 2024. We hope that others will contribute to improve, and extend, our codebase.

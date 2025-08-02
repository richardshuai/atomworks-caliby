# GPU-Accelerated MSA Generation

## Usage
The main script is launched from `make_msas_mmseqs_gpu.sh`.

It requires four arguments:

`--input_file`: the path to the input file containing the sequences to be aligned. (See details below)

`--output_dir`: the path to the output directory where the individual MSA files will be saved. This should be an empty directory.

`--sequence_col`: the name of the column in the input file that contains the sequences to be aligned.

`--chain_type_col`: the name of the column in the input file that contains the chain type of the sequences to be aligned. If this is not present in your dataframe it is ignored and assumed that all sequences are L-amino acid polypeptide chains.

### Input file requirements:

The script only expects a `.csv` or `.parquet` file with a single column of sequences and ignores all other columns. The sequences should be in the standard single-character protein alphabet with all non-standard residues as `X`.

If you're starting from a datahub output `pn_units` parquet file, you should be able to use this directly with no modifications.

### Output files:

The script outputs a single `.a3m` MSA file for each input sequence. By default the names of these files are simply the indices of the sequences in the input file (with duplicate sequences skipped).
* If you want to format the files for the AF3 pipeline, run `rename_msa_files_with_seq_has.py` on the output directory. This will rename the files according to the SHA-256 hash of their primary sequence and shard them into nested directories in the format that the AF3/datahub pipelines expect
* If you'd like to pair MSAs together using TaxID information (or allow the AF3 pipeline to pair on the fly), you need to also run `add_taxids_to_msa.py` on the output directory. This adds TaxID information to the header of each sequence in the MSA when available. Note that this requires ~30-40GB of memory to load the precomputed Uniref code <> TaxID map.

### Advanced options:

If you want to use more advanced options than those exposed in `make_msa_mmseqs_colabfold.py/run_mmseqs_pipeline()`, you can directly call the `make_msa_mmseqs_colabfold.py/_mmseqs_search_monomer()` function which has many more of the MMseqs2 parameters exposed. 

**Warning**: If you want to change the database you are searching against or are not using a GPU-DB node, be careful of IO-related issues. If you are doing one run once in a while with a non-local database, this should generally be fine in terms of digs network traffic. But problems could arise if you do this consistently, for a long time, or have many jobs like this running concurrently. Also, during testing I ran into nasty intermittent bugs (random segfaults hours into the runs) that only went away when I used a database stored on the local drive. You have been warned!

## Best Practices

* Use the GPU-DB node (`gpu37`)! This is the only way to both use a GPU and have the database stored locally which avoids nasty IO problems. The `make_msas_mmseqs_gpu.sh` script is already configured with the right `sbatch` options to use this node.
* Use large batches, but not too large. You can split your sequences across multiple input files and run separete jobs for each one. There's several minutes of overhead per batch, so if you split into 100 batches of 10 sequences that won't be efficient at all. A sweet spot seems to be around 1k-10k sequences per batch to get good performance without having to wait forever. The default `sbatch` configurations in `make_msas_mmseqs_gpu.sh` will allow up to 5 concurrent jobs with 2 GPUs each.
* If you're using these MSAs with datahub or AF3 pipelines, don't forget to run `rename_msa_files_with_seq_has.py` and `add_taxids_to_msa.py` as necessary.


## Performance

It depends on the length of the sequences and some other factors, but I've seen it take between 6 and 10 hours to generate 10k MSAs on 2 GPUs. So if you're running concurrent jobs on gpu37, you can probably generate on the order of 100k+ MSAs per day (assuming you're not waiting for the node to be available).

## Methods

This script uses MMseqs2 as described in the [Colabfold data preprocessing pipeline](https://www.nature.com/articles/s41592-022-01488-1). There's also a helpful [guide](https://github.com/soedinglab/mmseqs2/wiki) on how to use MMseqs2 on its own including the more recent [GPU support](https://www.biorxiv.org/content/10.1101/2024.11.13.623350v1).

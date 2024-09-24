# Datasets and Samplers Overview

This README provides an overview of how to handle, extend, and add new datasets for training structure-based models utilizing the `datahub` and `cifutils` repositories.
It's still a work-in-progress; additional contributions (to the codebase, AND the README) welcome - and encouraged!

## What is the high-level sampling and data loading workflow?

At a high-level, dataloading occurs through the following steps:
1. Sample a dataset index to load via a Sampler, like standard PyTorch. Dataset indices represent examples in a dataframe that we want featurize and pass to the model for training. For example, the index "1" in the dataset "PDB Chains" might correspond to PDB ID "3ne2", biassembly "2", PN Unit (similar to a chain, see the `Glossary` in the `cifutils` `README`) "A_1".
2. Call `__getitem__` on the top-level dataset associated with the Torch `DataLoader` using the dataset index (e.g., "1"). There may be multiple datasets concatenated together, so we must identify which dataset the example came from to call the appropriate `__getitem__` method to load the dataframe row.
3. Given that dataset index (e.g., "1"), call `__getitem__` on the corresponding sub-dataset to retrieve the relevant row. For example, our row could be a `pd.Series` containing values: `{"pdb_id": "3ne2", "q_pn_unit_iid": "A_1", "assembly": "2", "extra_info": "proteins4ever"}`
4. (Within `load_example_from_metadata_row`) Parse the dataframe row into a standardized format we can proceed with. Although often trivial, we must standardize the outputs of individual dataset `__getitem__` before proceeding, build the path to the CIF/PDB file, and aggregate the appropriate information. For our example, we may use the `PNUnitsDFParser` (which inherits the `ABC` `MetadataRowParser`) to parse our `pd.Series` row into a dictionary like `{"example_id": "{pn_unit}{3ne2}{2}{A_1}", "path": "/somewhere/on/digs/3ne2.cif", "q_pn_units": "A_1"}`. Only the `example_id` and the `path` are required; however, some transformations must receive additional fiels (`q_pn_units` for cropping, for example).
5. (Within `load_example_from_metadata_row`) Given the path provided in the standardized dictionary, load the CIF/PDB file with the `CIFParser`.
6. (Within `load_example_from_metadata_row`) Combine the output of the `CIFParser` with the output from the `PNUnitsDFParser.
7. Last, with our complete dictionary containing the output of the `CIFParser` and possibly additional fields from the `MetadataRowParser` (optional, depending on the `Transforms` used), execute the `Transforms` pipeline, and return the featurized results.

Whew, that was a lot of steps. Thankfully, most of that complexity is abstracted away and many users only need to concern themselves with small portions of the pipeline.

## How do I add a new dataset for training?

There are a few steps to add a new dataset for training.

### Step 1: Create the dataframe on disk.
First, we need a representation of the dataset on disk. I.e., one row for each "example" in the dataset. The rows can contain whatever information is needed - but most contain sufficient information such that we know where to find the CIF/PDB file, which assembly to load, and where to crop, if applicable.

For example, explore the a subset of the `interfaces` dataframe used to train AF-3-style models within `tests/data`, or the entire dataframe, including the whole PDB, at `/projects/ml/RF2_allatom/datasets/pdb/2024_09_10/interfaces_df.parquet` (NOTE: This file is large; you may want to limit the columns).

For the purposes of this example, let's say I stored a `parquet` file to `/digs/new_df.parquet` containing a `pdb_id` column and a `assembly` column. Let's also pretend that this row reference a file stored at `/databased/rcsb/pdb_id.cif`.

### Step 2: Write a `MetadataRowParser` for the dataframe.
As described in `datahub/datasets/dataframe_parsers`, we need `MetadataRowParsers` to convert the information held in a loaded row into a common format.

If there were no existing row parsers that met our needs, we could trivially write a new one:
```python
class InterfacesDFParser(MetadataRowParser):
    def __init__(self, base_dir: Path = Path("/databases/rcsb"), file_extension: str = ".cif"):
        self.base_dir = base_dir
        self.file_extension = file_extension
    
    def _parse(self, row: pd.Series) -> Dict[str, Any]:
        pdb_id = row["pdb_id"]
        assembly = row["assembly]
        path = Path(f"{self.base_dir}/{pdb_id[1:3]}/{pdb_id}{self.file_extension}")
        return {
            "example_id": f"{pdb_id}_{assembly}" # Must be unique within the dataframe
            "path": path,
            "assembly_id": row["assembly_id"],
        }
```

### Step 3: Add any templates or MSA's to the appropriate locations.
If using the standard MSA loading transforms, for MSAs to be correctly recognized, we must:
- Move the MSA's to `/projects/msa` (if others may find them useful; not required)
- Rename the MSA files according to the SHA-256 hash of the query sequence (we MUST use the function provided at `datahub.utils.misc.hash_sequence`) (the scripts in `scripts/preprocessing/msa` may be helpful)
- Ensure that the msa directory is a **flat** directory (again, the scripts in `scripts/preprocessing/msa` may be helpful)
- Add the directory path to the `msa_dirs` argument for `LoadPolymerMSAs` within the `hydra` configuration

For templates, we don't currently support any way to add additional sources.

### Step 4: Update the main training YAML.
Update the main training YAML with the new dataset (i.e., instantiate a `PandasDataset` wrapped within a `StructuralDatasetWrapper`), and you're good to go!

## How do we handle hierarchical dataset and sampling structures in Datahub?
We implemented a hierarchical dataset and sampling schema that can be extended indefinitely. Note that this setup is somewhat complex; most scenarios can be handled with a `StructuralDatasetWrapper` around a `PandasDataset`, or something similar.

### Example: Possible AF-3 Style Training Setup
For RF2AA, we have the following approximate dataset structure:

```
                 NamedConcatDataset
                        |
        ---------------------------------
        |                               |
 FB Distillation                NamedConcatDataset
(StructuralDatasetWrapper               |
 wrapping a PandasDataset)              |
                                        |
                                -----------------------
                                |                     |
                      Interfaces Dataset       PN Units Dataset
                    (StructuralDatasetWrapper  (StructuralDatasetWrapper
                     wrapping a PandasDataset)  wrapping a PandasDataset)
```

We may choose to implement a corresponding sampling schema like so:
```

                DistributedMixedSampler
                           |
                -------------------------
                |                       |
               0.2                     0.8
             Sampler1              MixedSampler
                                    /       \
                                   0.7       0.3
                                Sampler2   Sampler3

```
Then, with (0.8)(0.7) = (0.56) probability, we will draw an index corresponding to the Interfaces Dataset, with (0.2) probability we will draw from distillation, etc.

## Contributing
The initial version of this codebase was written by [Nate Corley](mailto:ncorley@uw.edu) and [Simon Mathis](mailto:simon.mathis@gmail.com) in the summer of 2024, in preparation for training an AF3-like structure prediction model. 
We only detail a skeleton that suffices for training RF2AA and AF-3; we hope that others will contribute to improve, and extend, our codebase.
# Unified Data Pre-Processing of All PDB Entries

This folder contains the pre-processing code required to build train, validation, and test splits for structure-based deep learning models. We handle all common entity types in the PDB: *proteins*, *rna*, *dna*, *hybrids*, and *small molecules*.
Pre-processing steps includes clash resolution, field extraction, and other data cleaning operations.
.

## Glossary
For details on our composable naming convention, please see the `Glossary` in the [cifutils](https://git.ipd.uw.edu/ai/cifutils#glossary) README.

## Pipeline
For details on the scripts needed to run the pipeline, see the `scripts` README in the [preprocessing](https://git.ipd.uw.edu/-/ide/project/ai/datahub/edit/main/-/datahub/preprocessing/) subfolder.
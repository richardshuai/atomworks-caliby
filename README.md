[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI version](https://img.shields.io/pypi/v/atomworks.svg)](https://pypi.org/project/atomworks/)
[![Python versions](https://img.shields.io/pypi/pyversions/atomworks.svg)](https://pypi.org/project/atomworks/)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://baker-laboratory.github.io/atomworks-dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

<img src="docs/_static/atomworks_logo.png" width="350" alt="atomworks logo">
# atomworks

**atomworks** is an open-source platform for next-generation biomolecular data processing, conversion, and machine-learning-ready featurization.  
It is composed of two symbiotic libraries:

- **atomworks.io:** A universal Python toolkit for parsing, cleaning, manipulating, and converting biological data (structures, sequences, small molecules). Built on the [biotite](https://www.biotite-python.org/) API, it seamlessly loads and exports between standards like mmCIF, PDB, FASTA, SMILES, MOL, and more.
- **atomworks.ml:** Advanced dataset featurization and sampling for deep learning workflows—using atomworks.io as its structural backbone.

The atomworks ecosystem is designed to eliminate the pain of file conversion and preprocessing, offering scientists and modelers an efficient, unified interface for biomolecular data.

---

## atomworks.io

*A swiss-army knife for biomolecular files in Python*

**atomworks.io** lets you:
- Parse, convert, and clean up any common biological file (structure or sequence).
- Transform all data to a consistent `AtomArray` representation for further analysis or machine learning.
- Model missing atoms, handle ligands/solvents, resolve naming/assembly heterogeneity—all from Python.

Instead of juggling dozens of tools or manual curation, simply load your data with atomworks.io and focus on your research.

---

## Installation

```
pip install atomworks # base installation version without torch (for only atomworks.io)
pip install "atomworks[ml]" # with torch and ML dependencies (for atomworks.io plus atomworks.ml)
pip install "atomworks[dev]" # with development dependencies
pip install "atomworks[ml,dev]" # with all dependencies

```

For more advanced setup options (including how to run workflows via apptainers) see the [full documentation](https://baker-laboratory.github.io/atomworks-dev/).

---

## Quick Start

```

from atomworks.io.parser import parse

result = parse(filename="3nez.cif.gz")

for chain_id, info in result["chain_info"].items():
print(chain_id, info["sequence"])

```

Output includes:
- **chain_info** — Sequences/metadata for each chain
- **ligand_info** — Ligand annotation & metrics
- **asym_unit** — Structure (AtomArrayStack)
- **assemblies** — Built biological assemblies
- **metadata** — Experimental and source information

See [usage examples](https://baker-laboratory.github.io/atomworks-dev/latest/usage/).

---

## When to use atomworks.io vs atomworks.ml?

- Use **atomworks.io** when you:
    - Need to parse/clean/convert between biological file formats (mmCIF, PDB, FASTA, etc.)
    - Want a unified structural representation to plug into any downstream analysis or modeling
    - Need structural operations like adding missing atoms, filtering ligands/solvents, or assembly generation

- Use **atomworks.ml** when you:
    - Need to featurize entire datasets for deep learning
    - Want ready-made sampling and batching utilities for training pipelines
    - Already use atomworks.io and want a seamless bridge to ML-ready feature engineering

---

## Contribution

We welcome improvements!  
Please see the [contributing guide](CONTRIBUTING.md) or [full documentation](https://baker-laboratory.github.io/atomworks-dev/).

---

## License

MIT License • Copyright (c) IPD, Baker Laboratory

---

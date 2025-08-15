[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI version](https://img.shields.io/pypi/v/atomworks.svg)](https://pypi.org/project/atomworks/)
[![Python versions](https://img.shields.io/pypi/pyversions/atomworks.svg)](https://pypi.org/project/atomworks/)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://baker-laboratory.github.io/atomworks-dev/latest)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

<img src="docs/_static/atomworks_logo_color.svg" width="450" alt="atomworks logo">

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

## atomworks.ml

*Advanced dataset featurization and sampling for deep learning workflows*

**atomworks.ml** provides:
- Ready-made featurization pipelines for entire datasets
- Efficient sampling and batching utilities for training machine learning models
- Seamless integration with atomworks.io for ML-ready feature engineering
- Optimized data structures and workflows designed specifically for deep learning applications

Built on atomworks.io's structural backbone, atomworks.ml bridges the gap between biological data processing and machine learning pipelines.

---

## Installation

```
pip install atomworks # base installation version without torch (for only atomworks.io)
pip install "atomworks[ml]" # with torch and ML dependencies (for atomworks.io plus atomworks.ml)
pip install "atomworks[dev]" # with development dependencies
pip install "atomworks[ml,dev]" # with all dependencies
```

If you are using [uv](https://docs.astral.sh/uv/reference/policies/versioning/) for package management, you can install atomworks with:

```
 uv pip install "atomworks[ml,openbabel,dev]"
```

For more advanced setup options (including how to run workflows via apptainers) see the [full documentation](https://baker-laboratory.github.io/atomworks-dev/latest).

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

See [usage examples](https://baker-laboratory.github.io/atomworks-dev/latest/auto_examples/).

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
Please see the [full documentation](https://baker-laboratory.github.io/atomworks-dev/latest) for contribution guidelines.

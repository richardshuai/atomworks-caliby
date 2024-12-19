[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pipeline status](https://github.com/baker-laboratory/datahub/actions/workflows/lint.yaml/badge.svg)](https://github.com/baker-laboratory/datahub/actions/workflows/lint.yaml)
[![pipeline status](https://github.com/baker-laboratory/datahub/actions/workflows/test.yaml/badge.svg)](https://github.com/baker-laboratory/datahub/actions/workflows/test.yaml)

# DataHub
Welcome to `DataHub` — our central data pipeline `Transform` library for training structure-based machine learning models.

DataHub enables you to go from `clean` structural data (e.g. the output of [cifutils](https://git.ipd.uw.edu/ai/cifutils) or an appropriately predicted computational structure)
to the inputs for a deep learning model. The codebase comprises various featurisation utils, logic to perform transforms on structural, MSA, Template and chemical
data, as well as utils for sampling and datasets to effectively train in a multi GPU or multi-GPU node setting.

# Usage instructions:
## 1. From an apptainer that does not have datahub installed:
The apptainer you are using will need to have the datahub dependencies installed. If that is the case and 
if you are at the IPD you can run:

```bash
# If cifutils is installed in the apptainer you're good.
# Else, you will need to add cifutils/src to your PYTHONPATH
source /path/to/cifutils/.ipd/setup.sh
# alternatively: export PYTHONPATH=/path/to/cifutils/src:$PYTHONPATH

source ./.ipd/setup.sh  # this will set your paths correctly for the IPD (it will put datahub in your PYTHONPATH)

# Then you can run the apptainer with datahub installed:
apptainer run /script/you/want/to/run.py
```

As an example, you can test the offical `.ipd/datahub.sif` apptainer with
```bash
apptainer exec .ipd/datahub.sif pytest tests/
```

# Overview

## How do I load a structural file?
Please see:
* The `cifutils` README, [here](https://github.com/baker-laboratory/cifutils). `cifutils` handles parsing of structural files, while `datahub` handles featurization.
* The simple exampled found in `examples/04_just_loading_pdbs_and_cifs.ipynb`

## How do I create a new dataset (e.g., distillation, fine-tuning, etc.)?
Please see:
* The [datasets](https://github.com/baker-laboratory/datahub/tree/main/datahub/datasets) README in `datahub/datasets`. Note that if your dataset consists of computationally-predicted files this pipeline may be dramatically simplified; e.g., the data cleaning and processing scripts may be unecessary. An example of a pipeline for a computationally predicted dataset is found in `/examples/02_adding_a_new_dataset.ipynb`; however, keep in mind that we no longer need to create a new `MetadataRowParser` for each dataset (we instead use the `GenericDFParser`).
* If you are creating a distillation set from non-predicted files that need a rigorous data processing pipeline, see the [preprocessing](https://github.com/baker-laboratory/datahub/tree/main/scripts/preprocessing) README in `/scripts/preprocessing`.

## How do I create a new `Transform`?
TBD... contibutions welcome!


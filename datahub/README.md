[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pipeline status](https://git.ipd.uw.edu/ai/datahub/badges/main/pipeline.svg)](https://git.ipd.uw.edu/ai/datahub/-/commits/main)
[![coverage report](https://git.ipd.uw.edu/ai/datahub/badges/main/coverage.svg)](https://git.ipd.uw.edu/ai/datahub/-/commits/main)

# DataHub
DataHub enables you to go from `clean` structural data (e.g. the output of [cifutils](https://git.ipd.uw.edu/ai/cifutils) or an appropriately predicted computational structure)
to the inputs to a deep learning model. It comprises various complex featurisation utils, logic to perform transforms on structural, MSA, Template and chemical
data as well as utils for sampling and datasets to effectively train in a multi GPU or multi-GPU node setting.

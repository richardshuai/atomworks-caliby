Extensions
==========

datahub: ML-Ready Data from cifutils
------------------------------------

The `datahub` library is an extension to cifutils that takes the output of cifutils (such as AtomArrays and metadata) and converts it into machine learning-ready tensors for deep learning workflows.

Typical workflow:

1. Use cifutils to parse and preprocess your structural data.
2. Pass the resulting AtomArray or dictionary to datahub for featurization and batching.

For more information, see the datahub repository:

- https://github.com/baker-laboratory/datahub 
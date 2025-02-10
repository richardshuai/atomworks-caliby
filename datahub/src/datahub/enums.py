"""Enums for datahub."""

from enum import IntEnum


class GroundTruthConformerPolicy(IntEnum):
    """Enum for ground truth conformer policy.

    Possible values are:
        -  REPLACE: Use the ground-truth coordinates as the reference conformer.
        -  FALLBACK: Use the ground-truth coordinates only if our standard conformer generation pipeline fails (e.g., we cannot generate a conformer with RDKit,
            and the molecule is either not in the CCD or the CCD entry is invalid).
        -  IGNORE: Do not use the ground-truth coordinates as the reference conformer under any circumstances.
    """

    REPLACE = 1
    FALLBACK = 2
    IGNORE = 3

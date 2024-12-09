import logging
import os

import numpy as np

# from cifutils.parser import CIFParser  # noqa: F401

# Set global logging level to `WARNING` if not set by user
logger = logging.getLogger("cifutils")
_log_level = os.environ.get("CIFUTILS_LOG_LEVEL", "WARNING").upper()
logger.setLevel(_log_level)

# Monkey patch biotite
import biotite.structure as struc  # noqa: E402

from cifutils.utils.selection_utils import get_residue_starts  # noqa: E402

struc.get_residue_starts = get_residue_starts


# TODO: Remove this patch with biotite 1.0.2 if #714 gets merged
def equal_annotations(self, item, equal_nan: bool = True):
    """
    Check, if this object shares equal annotation arrays with the
    given :class:`AtomArray` or :class:`AtomArrayStack`.

    Parameters
    ----------
    item : AtomArray or AtomArrayStack
        The object to compare the annotation arrays with.
    equal_nan: bool
        Whether to count `nan` values as equal. Default: True.

    Returns
    -------
    equality : bool
        True, if the annotation arrays are equal.
    """
    if not isinstance(item, struc.atoms._AtomArrayBase):
        return False
    if not self.equal_annotation_categories(item):
        return False
    for name in self._annot:
        # ... allowing `nan` values causes type-casting, which is only possible for floating-point arrays
        allow_nan = equal_nan if np.issubdtype(self._annot[name].dtype, np.floating) else False
        if not np.array_equal(
            self._annot[name],
            item._annot[name],
            equal_nan=allow_nan,
        ):
            return False
    return True


setattr(struc.atoms._AtomArrayBase, "equal_annotations", equal_annotations)

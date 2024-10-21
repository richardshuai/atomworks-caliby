from __future__ import annotations
from collections import OrderedDict
from typing import Any
import numpy as np
from toolz import reduce


def exists(obj: Any) -> bool:
    return obj is not None


def default(obj: Any, default: Any) -> Any:
    return obj if exists(obj) else default


def deduplicate_iterator(iterator):
    """Deduplicate an iterator while preserving order."""
    return iter(OrderedDict.fromkeys(iterator))


def to_hashable(element):
    return element if isinstance(element, (int, str, np.integer, np.str_)) else tuple(element)


def sum_string_arrays(*objs: np.ndarray | str) -> np.ndarray:
    """
    Sum a list of string arrays / strings into a single string array by concatenating them and
    determining the shortest string length to set as dtype.
    """
    return reduce(np.char.add, objs).astype(object).astype(str)


def not_isin(element: np.ndarray, array: np.ndarray, **isin_kwargs) -> np.ndarray:
    return np.isin(element, array, invert=True, **isin_kwargs)

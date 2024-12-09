from __future__ import annotations

import copy
from collections import OrderedDict
from functools import lru_cache, wraps
from typing import Any, Callable

import numpy as np
from toolz.curried import compose, reduce


def exists(obj: Any) -> bool:
    return obj is not None


def default(obj: Any, default: Any) -> Any:
    return obj if exists(obj) else default


def deduplicate_iterator(iterator):
    """Deduplicate an iterator while preserving order."""
    return iter(OrderedDict.fromkeys(iterator))


def to_hashable(element):
    """Convert an element to a hashable type."""
    return element if isinstance(element, (int, str, np.integer, np.str_)) else tuple(element)


def sum_string_arrays(*objs: np.ndarray | str) -> np.ndarray:
    """
    Sum a list of string arrays / strings into a single string array by concatenating them and
    determining the shortest string length to set as dtype.
    """
    return reduce(np.char.add, objs).astype(object).astype(str)


def not_isin(element: np.ndarray, array: np.ndarray, **isin_kwargs) -> np.ndarray:
    """Like `~np.isin`, but more efficient."""
    return np.isin(element, array, invert=True, **isin_kwargs)


def listmap(func: Callable, *iterables) -> list:
    """Like `map`, but returns a list instead of an iterator."""
    return compose(list, map)(func, *iterables)


def immutable_lru_cache(maxsize: int = 128, typed: bool = False):
    """An immutable version of `lru_cache` for caching functions that return mutable objects."""

    def decorator(func):
        cached_func = lru_cache(maxsize=maxsize, typed=typed)(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return copy.deepcopy(cached_func(*args, **kwargs))

        return wrapper

    return decorator

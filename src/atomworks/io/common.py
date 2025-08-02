from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from functools import lru_cache, wraps
from typing import Any

import numpy as np
from toolz.curried import compose, reduce


def exists(obj: Any) -> bool:
    return obj is not None


def default(obj: Any, default: Any) -> Any:
    return obj if exists(obj) else default


def deduplicate_iterator(iterator: Iterable) -> Iterator:
    """Deduplicate an iterator while preserving order."""
    return iter(OrderedDict.fromkeys(iterator))


def to_hashable(element: Any) -> Any:
    """Convert an element to a hashable type."""
    return element if isinstance(element, int | str | np.integer | np.str_) else tuple(element)


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


def immutable_lru_cache(maxsize: int = 128, typed: bool = False, deepcopy: bool = True) -> Callable:
    """An immutable version of `lru_cache` for caching functions that return mutable objects."""
    copy_func = copy.deepcopy if deepcopy else copy.copy

    def decorator(func: Callable) -> Callable:
        cached_func = lru_cache(maxsize=maxsize, typed=typed)(func)

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return copy_func(cached_func(*args, **kwargs))

        return wrapper

    return decorator


class KeyToIntMapper:
    """
    Maps keys to unique integers based on the order of the first appearance of the key.

    This is useful for mapping id's such as `chain_id`, `chain_entity`, `molecule_iid`, etc.
    to integers.

    Example:
    ```python
        chain_id_to_int = KeyToIntMapper()
        chain_id_to_int("A")  # 0
        chain_id_to_int("C")  # 1
        chain_id_to_int("A")  # 0
        chain_id_to_int("B")  # 2
    ```
    """

    def __init__(self):
        self.key_to_id = {}
        self.next_id = 0

    def __call__(self, value: Any) -> int:
        if value not in self.key_to_id:
            self.key_to_id[value] = self.next_id
            self.next_id += 1
        return self.key_to_id[value]


def md5_hash_string(s: str, length: int = 32) -> str:
    """Generate an MD5 hash of a string and return the first `length` characters."""
    full_hash = hashlib.md5(s.encode("utf-8")).hexdigest()
    return full_hash[:length]

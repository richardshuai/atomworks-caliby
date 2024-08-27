from __future__ import annotations
from collections import OrderedDict
from typing import Any
import numpy as np

def exists(obj: Any) -> bool:
    return obj is not None

def default(obj: Any, default: Any) -> Any:
    return obj if exists(obj) else default

def deduplicate_iterator(iterator):
    """Deduplicate an iterator while preserving order."""
    return iter(OrderedDict.fromkeys(iterator))

def to_hashable(element):
    return element if isinstance(element, (int, str, np.integer, np.str_)) else tuple(element)
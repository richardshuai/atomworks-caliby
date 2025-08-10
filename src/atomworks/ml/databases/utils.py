import json
from types import NoneType
from typing import Any, Literal, Union, get_args, get_origin

import numpy as np


def _smart_cast(value: Any, dtype: type) -> Any:
    """Cast a value to the correct type, handling special cases like and dicts (which are saved as json strings) and Optional / union types"""
    if (
        isinstance(dtype, dict) or get_origin(dtype) == dict  # noqa: E721
    ):
        return json.loads(value)
    elif get_origin(dtype) == Union:
        for arg in get_args(dtype):
            try:
                return _smart_cast(value, arg)
            except Exception:
                continue
        raise ValueError(f"Could not cast value {value} to any of the types in {dtype}")
    # with uppercase List[] we can smartly cast each element
    elif get_origin(dtype) == list:  # noqa: E721
        return [_smart_cast(str(v), get_args(dtype)[0]) for v in eval(value)]
    # with lower-case list we just eval and hope for the best
    elif dtype == list:  # noqa: E721
        return eval(value)
    elif dtype == Literal:
        return value
    elif dtype == NoneType and (value is None or np.isnan(value)):
        return None  # cast to None if the value is nan
    try:
        return dtype(value)
    except Exception:
        raise ValueError(f"Could not cast value {value} to {dtype}")  # noqa: B904

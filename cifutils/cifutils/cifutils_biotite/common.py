from __future__ import annotations
from typing import Any


def exists(obj: Any) -> bool:
    return obj is not None


def default(obj: Any, default: Any) -> Any:
    return obj if exists(obj) else default

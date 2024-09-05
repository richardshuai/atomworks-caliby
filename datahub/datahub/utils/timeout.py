"""
Timeout decorator. Adapted from https://github.com/pnpnpn/timeout-decorator/blob/master/timeout_decorator/timeout_decorator.py
    :copyright: (c) 2012-2013 by PN.
    :license: MIT, see LICENSE for more details.
"""

from __future__ import division, print_function, unicode_literals

import signal
from functools import wraps


class TimeoutError(AssertionError):
    """Thrown when a timeout occurs in the `timeout` context manager."""

    def __init__(self, value: str = "Timed Out"):
        self.value = value

    def __str__(self):
        return repr(self.value)


def _raise_exception(exception: Exception, exception_message: str | None = None):
    """Checks if a exception message is given.

    If there is no exception message, the default behaviour is maintained.
    If there is an exception message, the message is passed to the exception with the 'value' keyword.
    """
    if exception_message is None:
        raise exception()
    else:
        raise exception(exception_message)


def timeout_wrapper(
    seconds: float | None = None, timeout_exception: Exception = TimeoutError, exception_message: str | None = None
):
    """
    Add a timeout parameter to a function and return it.

    Args:
        - seconds (float, optional): Time limit in seconds or fractions of a second. If None, no timeout is applied.
            This adds flexibility to disable timing out depending on the settings.
        - timeout_exception (Exception): Exception to raise on timeout. Defaults to TimeoutError.
        - exception_message (str, optional): Message to pass to the exception on timeout.

    Returns:
        - Callable: The wrapped function with timeout functionality.

    Raises:
        - TimeoutError: If the time limit is reached.

    Note:
        It is illegal to pass anything other than a function as the first parameter. The function is wrapped and
        returned to the caller.
    """

    def decorate(function):
        def handler(*args, **kwargs):
            _raise_exception(timeout_exception, exception_message)

        @wraps(function)
        def new_function(*args, **kwargs):
            new_seconds = kwargs.pop("timeout", seconds)
            if new_seconds:
                old = signal.signal(signal.SIGALRM, handler)
                signal.setitimer(signal.ITIMER_REAL, new_seconds)

            if not seconds:
                return function(*args, **kwargs)

            try:
                return function(*args, **kwargs)
            finally:
                if new_seconds:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old)

        return new_function

    return decorate


class TimeoutContext:
    """
    A context manager for applying timeouts to blocks of code.

    Args:
        - seconds (float): Time limit in seconds or fractions of a second.
        - timeout_exception (Exception): Exception to raise on timeout. Defaults to TimeoutError.
        - exception_message (str, optional): Message to pass to the exception on timeout.

    Raises:
        - TimeoutError: If the time limit is reached.

    Example:
        with TimeoutContext(5):
            # Code that should timeout after 5 seconds
            time.sleep(10)
    """

    def __init__(
        self, seconds: float, timeout_exception: Exception = TimeoutError, exception_message: str | None = None
    ):
        self.seconds = seconds
        self.timeout_exception = timeout_exception
        self.exception_message = exception_message

    def _handler(self, signum, frame):
        _raise_exception(self.timeout_exception, self.exception_message)

    def __enter__(self):
        self.old_handler = signal.signal(signal.SIGALRM, self._handler)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.old_handler)

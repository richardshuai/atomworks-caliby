"""
Timeout utilities for applying time limits to functions and blocks of code.

Adapted from https://github.com/pnpnpn/timeout-decorator/blob/master/timeout_decorator/timeout_decorator.py (MIT License)
and from https://github.com/chaidiscovery/chai-lab/blob/main/chai_lab/utils/timeout.py
"""

from __future__ import division, print_function, unicode_literals

import multiprocessing
import queue as _queue
import signal
import time
from enum import Enum
from functools import wraps
from multiprocessing import Process, Queue
from typing import Any, Callable, Literal, assert_never


def timeout(default_timeout: float | int | None = None, strategy: Literal["signal", "subprocess"] = "signal"):
    """
    Decorator to apply a timeout to a function.

    The `signal` strategy is more efficient and slightly faster, but does not work in all contexts
    (e.g. with some C dependencies like RDKit, on certain operating systems).
    The `subprocess` strategy is always available, but slightly slower and with a higher overhead.
    """

    assert strategy in ["signal", "subprocess"]

    match strategy:
        case "signal":
            # timeout based on signal module
            return _build_timeout_using_singal_decorator(default_timeout)
        case "subprocess":
            # timeout based on subprocess module
            return _build_timeout_using_subprocess_decorator(default_timeout)
        case _:
            assert_never(strategy)


def _build_timeout_using_singal_decorator(default_timeout: float | int | None) -> Callable:
    """
    Build a decorator that applies a timeout to a function using the signal module.

    This decorator sets up a signal handler to raise a TimeoutError if the decorated function
    exceeds the specified timeout duration. It uses the SIGALRM signal to implement the timeout.

    Args:
        - default_timeout (float | int | None): The default timeout duration in seconds.

    Returns:
        - Callable: A decorator function that can be applied to other functions to add timeout functionality.
    """

    def decorate(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            # ... allow overriding the timeout with a keyword argument
            seconds = kwargs.pop("timeout", default_timeout)

            # no timeout
            if seconds is None:
                return func(*args, **kwargs)

            _start_time = time.time()

            def _timeout_handler(*_):
                # ... raise TimeoutError if called
                _elapsed_time = time.time() - _start_time
                raise TimeoutError(f"Function timed out after {_elapsed_time:.3f} seconds")

            # ... set the timeout handler and record the prior handler to restore later
            _prior_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            # ... start the timer
            signal.setitimer(signal.ITIMER_REAL, seconds)
            try:
                return func(*args, **kwargs)
            finally:
                # ... reset the timer
                signal.setitimer(signal.ITIMER_REAL, 0)
                # ... restore the prior handler
                signal.signal(signal.SIGALRM, _prior_handler)

        return wrapped_func

    return decorate


def _build_timeout_using_subprocess_decorator(default_timeout: float | int | None) -> Callable:
    """Force function to timeout after specified time.

    The returned decorator uses a subprocess to execute the function, allowing for timeout
    functionality even for CPU-bound operations that cannot be interrupted by signals.

    Args:
        timeout (float | int | None): The maximum time in seconds allowed for the function to execute.

    Returns:
        Callable: A decorator that can be applied to a function.

    Raises:
        TimeoutError: If the function does not return before the timeout.
        ChildProcessException: If the child process dies unexpectedly.
    """

    def _timeout_handler(queue: Queue, func: Callable, args: Any, kwargs: Any) -> None:
        try:
            result = func(*args, **kwargs)
            queue.put((_TimeoutHandlerStatus.SUCCESS, result))
        except Exception as e:
            queue.put((_TimeoutHandlerStatus.EXCEPTION, e))

    def decorator(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            # ... allow overriding the timeout with a keyword argument
            seconds = kwargs.pop("timeout", default_timeout)

            # no timeout
            if seconds is None:
                return func(*args, **kwargs)

            queue = Queue()

            # ... create subprocess to run the function
            proc = Process(target=_timeout_handler, args=(queue, func, args, kwargs), daemon=True)

            # ... start the subprocess (ensure it is not a daemon to allow doing this in multiprocessing)
            with _AllowSubprocessForDeamonicProcess():
                proc.start()

            # ... wait for the subprocess to finish and check if it timed out
            proc.join(timeout=float(seconds))

            # ... if the subprocess is still running, terminate it and raise a TimeoutError
            if proc.is_alive():
                proc.terminate()
                proc.join()
                raise TimeoutError(f"Function {func} timed out after {seconds} seconds")

            # ... try retrieving the result, if available
            try:
                status, value = queue.get(timeout=0.1)  # short timeout to prevent hang
                # NOTE: Hang can happen when the child process dies unexpectedly
                #       and the main process is waiting for the result in the queue.
                # See Issue(https://bugs.python.org/issue43805)
            except _queue.Empty:
                raise ChildProcessException("Child process died unexpectedly")

            match status:
                case _TimeoutHandlerStatus.SUCCESS:
                    # ... return the result of the function
                    return value
                case _TimeoutHandlerStatus.EXCEPTION:
                    # ... raise the exception caught in the child process
                    raise value
                case _:
                    # ... this code should be unreachable, if reached raise an error
                    assert_never(status)

        return wrapped_func

    return decorator


# TODO: This is dangerous: revert once the underlying problem in rdkit is fixed
# RDKit Issue(https://github.com/rdkit/rdkit/discussions/7289)
class _AllowSubprocessForDeamonicProcess(object):
    """Context Manager to resolve AssertionError: daemonic processes are not allowed to have children
    See https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic"""

    def __init__(self):
        self.conf: dict = multiprocessing.process.current_process()._config  # type: ignore
        if "daemon" in self.conf:
            self.daemon_status_set = True
        else:
            self.daemon_status_set = False
        self.daemon_status_value = self.conf.get("daemon")

    def __enter__(self):
        if self.daemon_status_set:
            del self.conf["daemon"]

    def __exit__(self, type, value, traceback):
        if self.daemon_status_set:
            self.conf["daemon"] = self.daemon_status_value


class _TimeoutHandlerStatus(Enum):
    """Status of the timeout handler."""

    SUCCESS = 0
    EXCEPTION = 1


class ChildProcessException(Exception):
    """Exception raised when a child process dies unexpectedly."""

    pass

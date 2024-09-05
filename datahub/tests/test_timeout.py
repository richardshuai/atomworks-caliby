import time

import pytest

from datahub.utils.timeout import TimeoutContext, TimeoutError, timeout_wrapper


def test_timeout_wrapper_no_timeout():
    @timeout_wrapper(seconds=2)
    def fast_function():
        return "Success"

    assert fast_function() == "Success"


def test_timeout_wrapper_with_timeout():
    @timeout_wrapper(seconds=0.1)
    def slow_function():
        time.sleep(0.5)
        return "This should not be returned"

    with pytest.raises(TimeoutError):
        slow_function()


def test_timeout_wrapper_custom_exception():
    class CustomTimeoutError(Exception):
        pass

    @timeout_wrapper(seconds=0.1, timeout_exception=CustomTimeoutError)
    def slow_function():
        time.sleep(0.5)

    with pytest.raises(CustomTimeoutError):
        slow_function()


def test_timeout_wrapper_custom_message():
    custom_message = "Custom timeout message"

    @timeout_wrapper(seconds=0.1, exception_message=custom_message)
    def slow_function():
        time.sleep(0.5)

    with pytest.raises(TimeoutError, match=custom_message):
        slow_function()


def test_timeout_context_no_timeout():
    with TimeoutContext(seconds=2):
        result = "Success"

    assert result == "Success"


def test_timeout_context_with_timeout():
    with pytest.raises(TimeoutError):
        with TimeoutContext(seconds=0.1):
            time.sleep(0.5)


def test_timeout_context_custom_exception():
    class CustomTimeoutError(Exception):
        pass

    with pytest.raises(CustomTimeoutError):
        with TimeoutContext(seconds=0.1, timeout_exception=CustomTimeoutError):
            time.sleep(0.5)


def test_timeout_context_custom_message():
    custom_message = "Custom timeout message"

    with pytest.raises(TimeoutError, match=custom_message):
        with TimeoutContext(seconds=0.1, exception_message=custom_message):
            time.sleep(0.5)


def test_timeout_wrapper_disable_timeout():
    @timeout_wrapper(seconds=None)
    def slow_function():
        time.sleep(0.5)
        return "Success"

    assert slow_function() == "Success"


def test_timeout_wrapper_override_timeout():
    @timeout_wrapper(seconds=0.1)
    def adjustable_function():
        time.sleep(0.5)
        return "Success"

    with pytest.raises(TimeoutError):
        adjustable_function()

    assert adjustable_function(timeout=1) == "Success"


if __name__ == "__main__":
    pytest.main(["-v", __file__])

import importlib
import logging
import os

import atomworks.io


def test_cifutils_logging_level():
    """
    Test if the logging level for cifutils is correctly configured.

    This test checks if the cifutils logger's level is set according to the CIFUTILS_LOG_LEVEL
    environment variable, or defaults to WARNING if not set.
    """
    # Get the expected level from the environment variable, default to WARNING
    expected_level_str = os.environ.get("CIFUTILS_LOG_LEVEL", "WARNING").upper()
    expected_level = getattr(logging, expected_level_str)

    # Get the cifutils logger
    cifutils_logger = logging.getLogger("cifutils")

    # Assert that the current logging level matches the expected level
    assert (
        cifutils_logger.level == expected_level
    ), f"Expected cifutils logging level to be {logging.getLevelName(expected_level)}, but it was {logging.getLevelName(cifutils_logger.level)}"


def test_cifutils_logging_level_env_var():
    """
    Test if the logging level for cifutils responds to changes in the CIFUTILS_LOG_LEVEL environment variable.
    """
    # Store the original environment variable value
    original_level = os.environ.get("CIFUTILS_LOG_LEVEL")

    try:
        # Set the environment variable
        os.environ["CIFUTILS_LOG_LEVEL"] = "DEBUG"

        # Re-import atomworks.io to trigger logger configuration
        importlib.reload(cifutils)

        # Get the cifutils logger
        cifutils_logger = logging.getLogger("cifutils")

        # Assert that the current logging level matches the set environment variable
        assert (
            cifutils_logger.level == logging.DEBUG
        ), f"Expected cifutils logging level to be DEBUG, but it was {logging.getLevelName(cifutils_logger.level)}"

    finally:
        # Clean up: restore the original environment variable
        if original_level is None:
            del os.environ["CIFUTILS_LOG_LEVEL"]
        else:
            os.environ["CIFUTILS_LOG_LEVEL"] = original_level

        # Reset the cifutils logger to its original state
        importlib.reload(cifutils)

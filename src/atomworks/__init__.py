"""
atomworks - Unified package for biological data I/O and machine learning.

This package combines functionality from atomworks.io (I/O operations) and atomworks.ml (ML utilities)
into a unified interface for biological data processing and machine learning.
"""

import logging
import os
import warnings

# Global logging configuration
logger = logging.getLogger("atomworks")
_log_level = os.environ.get("ATOMWORKS_LOG_LEVEL", "WARNING").upper()
logger.setLevel(_log_level)

# Ensure that deprecation warnings are not repeated
warnings.filterwarnings("once", category=DeprecationWarning)

# Apply monkey patching to extend AtomArray functionality
from atomworks.biotite_patch import monkey_patch_biotite  # noqa: E402

monkey_patch_biotite()


# Import version information
# Import subpackages
from . import io, ml  # noqa: E402

# Re-export key functionality from subpackages for convenience
# This maintains backward compatibility and provides a clean top-level API
# Key I/O functionality
from .io.parser import parse  # noqa: E402


def _get_versioning(repo_path: str) -> str:
    """
    Get the version string of the current git repository.

    The returned version string will be of the form:
        - <tag>[+dev{<commit_count>}.<commit_hash>][-dirty]

    [+dev{<commit_count>}.<commit_hash>] is only present if there are commits since
        the latest tag. The commit count is the number of commits between the latest
        tag and the current HEAD commit. The commit hash is the first 7 characters
        of the commit hash.
    [-dirty] is only present if there are uncommitted changes.
    """
    os.environ["GIT_PYTHON_REFRESH"] = "silent"
    import git

    repo = git.Repo(repo_path, search_parent_directories=True)

    # (1) Get latest tag
    latest_tag = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)[-1]
    version = str(latest_tag)

    # (2) Check for commits since latest tag
    if latest_tag.commit != repo.head.commit:
        # Count commits between latest tag and HEAD
        commit_count = len(list(repo.iter_commits(f"{latest_tag.commit}..HEAD")))
        version += f"+dev{commit_count}.{repo.head.commit.hexsha[:7]}"

    # (3) Check for uncommitted changes
    if repo.is_dirty(untracked_files=True):
        version += "-dirty"

    return version


# Import version information
__version__ = _get_versioning(os.path.dirname(os.path.realpath(__file__)))
assert __version__, "Failed to determine package version"

__all__ = [
    "__version__",
    "io",
    "ml",
    "monkey_patch_atomarray",
    "parse",
]

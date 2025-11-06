"""MolProbity executable wrapper for structure validation via cctbx."""

import logging
import os
from os import PathLike

from atomworks.ml.executables import Executable, ExecutableError

logger = logging.getLogger(__name__)


class MolProbity(Executable):
    """Executable wrapper for MolProbity validation script using cctbx.

    MolProbity provides comprehensive structure validation including Ramachandran
    analysis, rotamer validation, clash detection, and geometric validation.

    The molprobity script uses the cctbx Python interpreter which has built-in
    access to the required chem_data directory (rotamer library, etc.).

    Examples:
      >>> molprobity = MolProbity.get_or_initialize()
      >>> bin_path = molprobity.get_bin_path()
    """

    name: str = "molprobity"
    version_cmd: str = ""  # Custom Python script, no standard version command
    verification_cmd: str = ""  # No help command
    required_verification_text: tuple[str, ...] = ()  # Manual verification only

    @classmethod
    def initialize(cls, bin_path: PathLike | None = None, *args, **kwargs) -> "MolProbity":
        """Initialize MolProbity executable.

        Args:
          bin_path: Path to molprobity script. If ``None``, uses ``MOLPROBITY_PATH`` env variable.

        Returns:
          Initialized MolProbity executable.

        Raises:
          ExecutableError: If executable not found.
        """
        if bin_path is None:
            bin_path = cls._infer_bin_path_from_env_var()
        return super().initialize(bin_path, *args, **kwargs)

    @staticmethod
    def _infer_bin_path_from_env_var() -> PathLike:
        """Get molprobity script path from environment variables."""
        molprobity_path = os.environ.get("MOLPROBITY_PATH")
        if molprobity_path is not None and os.path.isfile(molprobity_path) and os.access(molprobity_path, os.X_OK):
            return molprobity_path

        raise ExecutableError(
            "No `bin_path` provided and `MOLPROBITY_PATH` environment variable not set.\n"
            "Please set the `MOLPROBITY_PATH` environment variable to the path of the molprobity script "
            "or provide a `bin_path` when initializing: "
            "`MolProbity.initialize(bin_path='/path/to/molprobity')`."
        )

    @classmethod
    def _setup(cls, bin_path: PathLike, *args, **kwargs) -> None:
        """Setup method for MolProbity (no special setup required).

        The molprobity script uses the cctbx Python interpreter which already
        has access to the chem_data directory.
        """
        pass

"""Global Settings module."""

import logging
import subprocess
from pathlib import Path

from pqueens.schedulers.scheduler import SHUTDOWN_CLIENTS
from pqueens.utils.logger_settings import setup_basic_logging
from pqueens.utils.path_utils import PATH_TO_QUEENS

_logger = logging.getLogger(__name__)
GLOBAL_SETTINGS = None


class GlobalSettings:
    """Class for global settings in Queens.

    Attributes:
        experiment_name (str): Experiment name of queens run
        output_dir (Path): Output directory for queens run
        git_hash (str): Hash of active git commit
        debug (bool): True if debug mode is to be used
    """

    def __init__(self, experiment_name, output_dir, debug=False):
        """Initialize global settings.

        Args:
            experiment_name (str): Experiment name of queens run
            output_dir (str, Path): Output directory for queens run
            debug (bool): True if debug mode is to be used
        """
        output_dir = Path(output_dir)
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Output directory {output_dir} does not exist.")

        # Remove spaces as they can cause problems later on
        if " " in experiment_name:
            raise ValueError("Experiment name can not contain spaces!")

        try:
            git_hash = (
                subprocess.check_output(
                    ['cd', f'{PATH_TO_QUEENS}', ';', 'git', 'rev-parse', 'HEAD']
                )
                .decode('ascii')
                .strip()
            )
        except subprocess.CalledProcessError as subprocess_error:
            _logger.warning("Could not get git hash. Failed with the following error:")
            _logger.warning(str(subprocess_error))
            git_hash = "unknown"
            _logger.warning("Setting git hash to: '%s'!", git_hash)

        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.git_hash = git_hash
        self.debug = debug

    def __enter__(self):
        """'enter'-function in order to use the global settings as a context.

        This function is called prior to entering the context.

        Returns:
            self
        """
        global GLOBAL_SETTINGS  # pylint: disable=global-statement
        GLOBAL_SETTINGS = self

        # set up logging
        log_file_path = self.output_dir / f'{self.experiment_name}.log'
        setup_basic_logging(log_file_path=log_file_path, debug=self.debug)

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """'exit'-function in order to use the global settings as a context.

        This function is called at the end of the context.

        The exception as well as traceback arguments are required to implement the `__exit__`
        method, however, we do not use them explicitly.

        Args:
            exception_type: indicates class of exception (e.g. ValueError)
            exception_value: indicates exception instance
            traceback: traceback object
        """
        global GLOBAL_SETTINGS  # pylint: disable=global-statement
        GLOBAL_SETTINGS = None

        for shutdown_client in SHUTDOWN_CLIENTS.copy():
            SHUTDOWN_CLIENTS.remove(shutdown_client)
            shutdown_client()

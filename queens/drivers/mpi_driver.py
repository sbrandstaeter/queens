"""Convenience wrapper around Jobscript Driver."""

from functools import partial

from queens.drivers.jobscript_driver import JobscriptDriver
from queens.utils.path_utils import relative_path_from_queens

MpiDriver = partial(
    JobscriptDriver,
    jobscript_template=relative_path_from_queens("templates/jobscripts/jobscript_local.sh"),
)

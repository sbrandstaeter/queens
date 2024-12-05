#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Configuration of folder structure of QUEENS experiments."""

import logging
from pathlib import Path

from queens.utils.path_utils import create_folder_if_not_existent

_logger = logging.getLogger(__name__)

BASE_DATA_DIR = "queens-simulation-data"
EXPERIMENTS_BASE_FOLDER_NAME = "experiments"
TESTS_BASE_FOLDER_NAME = "tests"


def base_directory():
    """Hold all queens related data."""
    base_dir = Path().home() / BASE_DATA_DIR
    create_directory(base_dir)
    return base_dir


def experiments_base_directory():
    """Hold all experiment data on the computing machine."""
    base_dir = base_directory()
    experiments_base_dir = base_dir / EXPERIMENTS_BASE_FOLDER_NAME
    create_directory(experiments_base_dir)
    return experiments_base_dir


def experiment_directory(experiment_name):
    """Directory for data of a specific experiment on the computing machine.

    Args:
        experiment_name (str): Experiment name
    """
    experiments_base_dir = experiments_base_directory()
    experiment_dir = experiments_base_dir / experiment_name
    create_directory(experiment_dir)
    return experiment_dir


def create_directory(dir_path):
    """Create a directory either local or remote."""
    _logger.debug("Creating folder %s.", dir_path)
    create_folder_if_not_existent(dir_path)


def current_job_directory(experiment_dir, job_id):
    """Directory of the latest submitted job.

    Args:
        experiment_dir (Path): Experiment directory
        job_id (str): Job ID of the current job

    Returns:
        job_dir (Path): Path to the current job directory.
    """
    job_dir = experiment_dir / str(job_id)
    return job_dir


def job_dirs_in_experiment_dir(experiment_dir):
    """Get job directories in experiment_dir.

    Args:
        experiment_dir (pathlib.Path, str): Path with the job dirs

    Returns:
        job_directories (list): List with job_dir paths
    """
    experiment_dir = Path(experiment_dir)
    job_directories = []
    for job_directory in experiment_dir.iterdir():
        if job_directory.is_dir() and job_directory.name.isdigit():
            job_directories.append(job_directory)

    # Sort the jobs directories
    return sorted(job_directories, key=lambda x: int(x.name))

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
"""Unit tests for the jobscript driver."""

import numpy as np
import pytest
import yaml

from queens.distributions import FreeVariable
from queens.drivers.jobscript_driver import JobOptions, JobscriptDriver
from queens.parameters import Parameters


@pytest.fixture(name="parameters")
def fixture_parameters():
    """Parameters for the jobscript driver test."""
    parameters = Parameters(parameter_1=FreeVariable(1), parameter_2=FreeVariable(1))
    return parameters


@pytest.fixture(name="job_id")
def fixture_job_id():
    """Fixture for the job id."""
    return 42


@pytest.fixture(name="experiment_name")
def fixture_experiment_name():
    """Fixture for the experiment_name."""
    return "test_experiment"


@pytest.fixture(name="job_options")
def fixture_job_options(tmp_path, job_id, experiment_name, injected_input_files):
    """Job options to be injected."""
    num_procs = 4
    experiment_dir = tmp_path

    job_options = JobOptions(
        job_dir=tmp_path / str(job_id),
        output_dir=tmp_path / str(job_id) / "output",
        output_file=tmp_path / str(job_id) / f"output/{experiment_name}_{job_id}",
        job_id=job_id,
        num_procs=num_procs,
        experiment_dir=experiment_dir,
        experiment_name=experiment_name,
        input_files=injected_input_files,
    )

    return job_options


@pytest.fixture(name="input_templates")
def fixture_input_templates(tmp_path, job_options, parameters):
    """Generate input templates."""

    def templatify(list_of_keys, template_path):
        """Create dict from list of keys."""
        template_dict = {f: "{{ " + f + " }}" for f in list_of_keys}
        # yaml format is used to ease read in of input file
        template_path.write_text(yaml.safe_dump(template_dict))

    # only add the second parameter
    input_template_1 = tmp_path / "input_template_1.yaml"
    templatify(list(job_options.to_dict().keys()) + parameters.names[-1:], input_template_1)

    # add both parameters to the template
    input_template_2 = tmp_path / "input_template_2.yaml"
    templatify(parameters.names + ["input_1", "experiment_name"], input_template_2)

    return input_template_1, input_template_2


@pytest.fixture(name="injected_input_files")
def fixture_injected_input_files(tmp_path, job_id, experiment_name):
    """Fixture for the create input files."""
    input_file_1 = tmp_path / str(job_id) / f"{experiment_name}_input_1_{job_id}.yaml"
    input_file_2 = tmp_path / str(job_id) / f"{experiment_name}_input_2_{job_id}.yaml"
    injected_input_files = {"input_1": input_file_1, "input_2": input_file_2}

    return injected_input_files


@pytest.fixture(name="jobscript_driver")
def fixture_jobscript_driver(parameters, input_templates):
    """Jobscript driver object."""
    input_template_1, input_template_2 = input_templates

    driver = JobscriptDriver(
        parameters=parameters,
        jobscript_template="",
        executable="",
        input_templates={"input_1": input_template_1, "input_2": input_template_2},
    )

    return driver


def test_jobscript_driver_multiple_input_files(
    jobscript_driver, job_options, injected_input_files, parameters
):
    """Test if multiple input files are correctly generated."""
    # Samples to be injected
    sample_dict = parameters.sample_as_dict(np.array([1, 2]))

    # Arguments to call the run method of the driver
    job_id = job_options.job_id
    num_procs = job_options.num_procs
    experiment_dir = job_options.experiment_dir
    experiment_name = job_options.experiment_name

    # Run the driver
    jobscript_driver.run(
        sample=np.array(list(sample_dict.values())),
        job_id=job_id,
        num_procs=num_procs,
        experiment_dir=experiment_dir,
        experiment_name=experiment_name,
    )

    # Join all options
    injectable_options = job_options.add_data_and_to_dict(sample_dict)

    # Check if all the data was injected correctly in all input files
    for input_file in injected_input_files.values():
        for key, value in yaml.safe_load(input_file.read_text()).items():
            assert value == str(injectable_options[key])

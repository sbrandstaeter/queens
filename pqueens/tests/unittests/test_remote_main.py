"""
Test if the remote main function catches failed singularity runs
"""

import json
import os

import pqueens
import pytest
from pqueens.remote_main import main


@pytest.fixture(scope="module", params=[True, False])
def finalize_fail(request):
    # This fixture selects if the driver patch fails when writing to the db
    return request.param


@pytest.fixture(scope="module", params=["000", "27017"])
def port(request):
    return request.param


@pytest.mark.unit_tests
def test_exit_conditions_remote_main(mocker, monkeypatch, finalize_fail, port):

    # Patch open() function
    mocker.patch("builtins.open", create=True)

    # Patch os.path.join function
    def os_path_join_response(*args, **kwargs):
        return "dummy"

    monkeypatch.setattr(os.path, "join", os_path_join_response)

    # Patch json.load function
    def json_load_response(*args, **kwargs):
        dummy_dict = {
            "experiment_name": "entry",
            "database": {"type": "mongodb", "name": "test_remote_main",},
            "scheduler": {"singularity_settings": {"remote_ip": "localhost"}},
        }
        return dummy_dict

    monkeypatch.setattr(json, "load", json_load_response)

    # Mock driver class
    class driver_mock:
        def __init__(self, raise_error_in_finalize):
            self.raise_error_in_finalize = raise_error_in_finalize

        def pre_job_run_and_run_job(self):
            raise ValueError("Mock singularity error")

        def post_job_run(self):
            raise ValueError("Mock singularity error")

        def finalize_job_in_db(self):
            if self.raise_error_in_finalize:
                raise ValueError(f"Mock finalize_job_in_db error")

    def driver_mock_response(*args, **kwargs):
        return driver_mock(finalize_fail)

    monkeypatch.setattr(
        pqueens.drivers.driver.Driver, "from_config_create_driver", driver_mock_response
    )

    # Dummy arguments
    args = [
        "--job_id=2",
        "--batch=10",
        "--port=" + port,
        "--path_json=dummypath",
        "--post=true",
        "--workdir=dummy_workdir",
        "--hash=false",
    ]

    if finalize_fail:
        error = "Mock finalize_job_in_db error"
    else:
        error = "Mock singularity error"

    with pytest.raises(ValueError) as excinfo:
        # Call the remote main
        main(args)

    # Assert if the correct error was thrown
    assert str(excinfo.value) == error

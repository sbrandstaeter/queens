"""Test if the remote main function catches failed singularity runs."""


import pytest

from pqueens.singularity.remote_main import main


@pytest.fixture(scope="module", params=[True, False])
def finalize_fail(request):
    """Fixture to test failure."""
    # This fixture selects if the driver patch fails when writing to the db
    return request.param


@pytest.fixture(scope="module", params=["000", "27017"])
def port(request):
    """Port for the singularity runs."""
    return request.param


def test_exit_conditions_remote_main(mocker, finalize_fail, port):
    """Test if an error is raised."""
    if finalize_fail:
        error = "Mock finalize_job_in_db error"
    else:
        error = "Mock singularity error"

    dummy_dict = {
        "experiment_name": "entry",
        "database": {
            "type": "mongodb",
            "name": "test_remote_main",
        },
        "scheduler": {"singularity_settings": {"remote_ip": "localhost"}},
    }

    mocker.patch(
        'pqueens.singularity.remote_main.get_config_dict',
        return_value=dummy_dict,
    )

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

    mocker.patch(
        'pqueens.singularity.remote_main.from_config_create_driver',
        return_value=driver_mock(finalize_fail),
    )

    mocker.patch(
        'pqueens.singularity.remote_main.DB_module.from_config_create_database',
    )

    mocker.patch("pqueens.singularity.remote_main.DB_module.database")

    # Dummy arguments
    args = [
        "--job_id=2",
        "--batch=10",
        "--port=" + port,
        "--path_json=dummypath",
        "--post=true",
        "--workdir=dummy_workdir",
    ]

    with pytest.raises(ValueError) as excinfo:
        # Call the remote main
        main(args)

    # Assert if the correct error was thrown
    assert str(excinfo.value) == error

"""Test remote BACI simulations with ensight data-processor."""

import logging
from pathlib import Path

import numpy as np
import pytest

import queens.schedulers.cluster_scheduler as cluster_scheduler  # pylint: disable=consider-using-from-import
from queens.data_processor.data_processor_ensight import DataProcessorEnsight
from queens.distributions.uniform import UniformDistribution
from queens.drivers.mpi_driver import MpiDriver
from queens.external_geometry.baci_dat_geometry import BaciDatExternalGeometry
from queens.interfaces.job_interface import JobInterface
from queens.iterators.monte_carlo_iterator import MonteCarloIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.schedulers.local_scheduler import LocalScheduler
from queens.utils import config_directories, io_utils
from queens.utils.fcc_utils import from_config_create_object
from queens.utils.io_utils import load_result
from tests.integration_tests.conftest import (  # BRUTEFORCE_CLUSTER_TYPE,
    CHARON_CLUSTER_TYPE,
    THOUGHT_CLUSTER_TYPE,
)

_logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "cluster",
    [
        pytest.param(THOUGHT_CLUSTER_TYPE, marks=pytest.mark.lnm_cluster),
        # pytest.param(BRUTEFORCE_CLUSTER_TYPE, marks=pytest.mark.lnm_cluster),
        pytest.param(CHARON_CLUSTER_TYPE, marks=pytest.mark.imcs_cluster),
    ],
    indirect=True,
)
class TestDaskCluster:
    """Test class collecting all test with Dask jobqueue clusters and Baci.

    NOTE: we use a class here since our fixture are set to autouse, but we only want to call them
    for these tests.
    """

    @pytest.fixture(autouse=True)
    def mock_experiment_dir(self, monkeypatch, cluster_settings, patched_base_directory):
        """Mock the experiment directory on the cluster.

        The goal is to separate the testing data from production data of the user.
        NOTE: It is necessary to mock the whole experiment_directory method.
        Otherwise, the mock is not loaded properly remote.
        This is in contrast to the local mocking where it suffices to mock
        config_directories.EXPERIMENTS_BASE_FOLDER_NAME.
        Note that we also rely on this local mock here!
        """

        def patch_experiments_directory(experiment_name):
            """Base directory for all experiments on the computing machine."""
            experiments_dir = (
                Path(patched_base_directory.replace("$HOME", str(Path.home()))) / experiment_name
            )
            Path.mkdir(experiments_dir, parents=True, exist_ok=True)
            return experiments_dir

        monkeypatch.setattr(cluster_scheduler, "experiment_directory", patch_experiments_directory)
        _logger.debug("Mocking of dask experiment_directory  was successful.")
        _logger.debug(
            "dask experiment_directory is mocked to '%s/<experiment_name>' on %s@%s",
            patched_base_directory,
            cluster_settings["user"],
            cluster_settings["host"],
        )

    def experiment_dir_on_cluster(self):
        """Remote experiment path."""
        return (
            f"$HOME/{config_directories.BASE_DATA_DIR}/{config_directories.TESTS_BASE_FOLDER_NAME}"
        )

    @pytest.fixture(name="patched_base_directory")
    def fixture_patched_base_directory(self, pytest_id):
        """Path of the tests on the remote cluster."""
        return self.experiment_dir_on_cluster() + f"/{pytest_id}"

    def test_baci_mc_ensight_cluster(
        self,
        tmp_path,
        third_party_inputs,
        cluster_settings,
        baci_cluster_paths,
        baci_example_expected_mean,
        baci_example_expected_var,
        baci_example_expected_output,
        _initialize_global_settings,
    ):
        """Test remote BACI simulations with DASK jobqueue and MC iterator.

        Test for remote BACI simulations on a remote cluster in combination
        with
        - DASK jobqueue cluster
        - Monte-Carlo (MC) iterator
        - BACI ensight data-processor.


        Args:
            self:
            inputdir (Path): Path to the JSON input file
            tmp_path (Path): Temporary directory for this test
            third_party_inputs (str): Path to the BACI input files
            cluster_settings (dict): Cluster settings
            baci_cluster_paths (dict): collection of paths to BACI executables on the cluster
            baci_example_expected_mean (np.ndarray): Expected mean for the MC samples
            baci_example_expected_var (np.ndarray): Expected var for the MC samples
            baci_example_expected_output (np.ndarray): Expected output for the MC samples
            patched_base_directory (str): directory of the test simulation data on the cluster
        """
        cluster_name = cluster_settings["name"]

        # unique experiment name
        experiment_name = f"baci_mc_ensight_{cluster_name}"

        baci_input_file_template = Path(
            third_party_inputs, "baci", "meshtying3D_patch_lin_duallagr_new_struct.dat"
        )

        # A main run is needed as you are using dask
        def run(_initialize_global_settings):
            # Parameters
            nue = UniformDistribution(lower_bound=0.4, upper_bound=0.49)
            young = UniformDistribution(lower_bound=500, upper_bound=1000)
            parameters = Parameters(nue=nue, young=young)

            # Setup QUEENS stuff
            external_geometry = BaciDatExternalGeometry(
                list_geometric_sets=["DSURFACE 1"], input_template="baci_input"
            )
            data_processor = DataProcessorEnsight(
                file_name_identifier="baci_mc_ensight_*structure.case",
                file_options_dict={
                    "delete_field_data": False,
                    "geometric_target": ["geometric_set", "DSURFACE 1"],
                    "physical_field_dict": {
                        "vtk_field_type": "structure",
                        "vtk_array_type": "point_array",
                        "vtk_field_label": "displacement",
                        "field_components": [0, 1, 2],
                    },
                    "target_time_lst": ["last"],
                },
                external_geometry=external_geometry,
            )
            scheduler = LocalScheduler(
                num_procs=2,
                num_procs_post=1,
                max_concurrent=2,
                global_settings=_initialize_global_settings,
            )
            driver = MpiDriver(
                input_template="baci_input",
                path_to_executable="baci_release",
                path_to_postprocessor="post_ensight",
                post_file_prefix="baci_mc_ensight",
                data_processor=data_processor,
            )
            interface = JobInterface(scheduler=scheduler, driver=driver, parameters=parameters)
            model = SimulationModel(interface=interface)
            iterator = MonteCarloIterator(
                seed=42,
                num_samples=2,
                result_description={"write_results": True, "plot_results": False},
                model=model,
                parameters=parameters,
                global_settings=_initialize_global_settings,
            )

            # Actual analysis
            run_iterator(iterator, _initialize_global_settings)

            # Load results
            result_file = tmp_path / "dummy_experiment_name.pickle"
            results = load_result(result_file)

        # main run
        if __name__ == "__main__":
            run(_initialize_global_settings)

        # Load results
        result_file = tmp_path / "dummy_experiment_name.pickle"
        results = load_result(result_file)

        # assert statements
        np.testing.assert_array_almost_equal(results['mean'], baci_example_expected_mean, decimal=6)
        np.testing.assert_array_almost_equal(results['var'], baci_example_expected_var, decimal=6)
        np.testing.assert_array_almost_equal(
            results['raw_output_data']['result'], baci_example_expected_output, decimal=6
        )

    def delete_simulation_data(self, input_file_path):
        """Delete simulation data on the cluster.

        This approach deletes test simulation data older then 7 days
        Args:
            input_file_path (pathlib.Path): Path to input file
        """
        # Create a remote connection
        remote_connection_option = io_utils.load_input_file(input_file_path)["my_remote_connection"]
        remote_connection = from_config_create_object(remote_connection_option)
        remote_connection.open()

        # Delete data from tests older then 1 week
        command = (
            "find "
            + str(self.experiment_dir_on_cluster())
            + " -mtime +7 -mindepth 1 -maxdepth 1 -type d -exec rm -rv {} \\;"
        )
        result = remote_connection.run(command, in_stream=False)
        _logger.debug("Deleting old simulation data:\n%s", result.stdout)

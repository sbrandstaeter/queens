"""Collect fixtures used by the cluster tests."""
import pytest

from pqueens.utils.path_utils import relative_path_from_queens

deep_cluster_settings = {
    'name': 'deep',
    'workload_manager': 'pbs',
    'cluster_address': '129.187.58.20',
    'cluster_python_path': '$HOME/anaconda/miniconda/envs/queens_p310/bin/python',
    'path_to_jobscript': relative_path_from_queens('templates/jobscripts/jobscript_deep.sh'),
}

bruteforce_cluster_settings = {
    'name': 'bruteforce',
    'workload_manager': 'slurm',
    'cluster_address': '10.10.0.1',
    'cluster_python_path': '$HOME/anaconda/miniconda/envs/queens_p310/bin/python',
    'path_to_jobscript': relative_path_from_queens('templates/jobscripts/jobscript_bruteforce.sh'),
}

charon_cluster_settings = {
    'name': 'charon',
    'workload_manager': 'slurm',
    'cluster_address': '192.168.1.253',
    'cluster_python_path': '$HOME/anaconda/miniconda/envs/queens_p310/bin/python',
    'path_to_jobscript': relative_path_from_queens('templates/jobscripts/jobscript_charon.sh'),
}


@pytest.fixture(params=[deep_cluster_settings])
def dask_cluster_settings(request):
    return request.param

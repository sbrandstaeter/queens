"""Settings used by the cluster tests."""

from pqueens.utils.path_utils import relative_path_from_queens

deep_cluster_settings = {
    'name': 'deep',
    'workload_manager': 'pbs',
    'cluster_address': 'deep.lnm.ed.tum.de',
    'cluster_internal_address': 'null',
    'cluster_python_path': '$HOME/anaconda/miniconda/envs/queens/bin/python',
    'path_to_jobscript': relative_path_from_queens('templates/jobscripts/jobscript_dask_deep.sh'),
    'cluster_script_path': '/lnm/share/donottouch.sh',
}

bruteforce_cluster_settings = {
    'name': 'bruteforce',
    'workload_manager': 'slurm',
    'cluster_address': 'bruteforce.lnm.ed.tum.de',
    'cluster_internal_address': '10.10.0.1',
    'cluster_python_path': '$HOME/anaconda/miniconda/envs/queens/bin/python',
    'path_to_jobscript': relative_path_from_queens(
        'templates/jobscripts/jobscript_dask_bruteforce.sh'
    ),
    'cluster_script_path': '/lnm/share/donottouch.sh',
}

charon_cluster_settings = {
    'name': 'charon',
    'workload_manager': 'slurm',
    'cluster_address': 'charon.bauv.unibw-muenchen.de',
    'cluster_internal_address': '192.168.1.253',
    'cluster_python_path': '$HOME/miniconda3/envs/queens/bin/python',
    'path_to_jobscript': relative_path_from_queens('templates/jobscripts/jobscript_dask_charon.sh'),
    'cluster_script_path': 'null',
}

"""Helper functions for handling clusters."""
import numpy as np


def get_cluster_job_id(scheduler_type, output_str_cluster, pbs_scheduler_types):
    """Retrieve id of a job after submitting it to the cluster.

    Args:
        scheduler_type (str): Type of cluster scheduler
        output_str_cluster (string): Output returned when submitting the job
        pbs_scheduler_types (list): List of valid schedulers with pbs

    Returns:
         job_id_return (str/None): job ID return value
    """
    if output_str_cluster:
        if scheduler_type in pbs_scheduler_types:
            job_cluster_id_return = int(output_str_cluster.split('.')[0])
        else:
            job_cluster_id_return = int(output_str_cluster.split()[-1])
    else:
        job_cluster_id_return = None

    return job_cluster_id_return


def distribute_procs_on_nodes_pbs(num_procs=1, max_procs_per_node=16):
    """Compute the distribution of processors on nodes of a PBS cluster.

    Args:
        num_procs (int): total number of requested processors
        max_procs_per_node (int):maximum number of processors of a node on the cluster

    Returns:
         - num_nodes (int): number of nodes needed for the job
         - procs_per_node (int): number of processors needed on each node
    """
    num_nodes = int(np.ceil(num_procs / max_procs_per_node))
    if num_procs % num_nodes == 0:
        procs_per_node = int(num_procs / num_nodes)
    else:
        raise ValueError(
            "An even distribution of processors on the nodes is required by the PBS scheduler!\n"
            f"Requested number of processors {num_procs} cannot be distributed evenly"
            f" on {num_nodes} nodes."
        )

    return num_nodes, procs_per_node

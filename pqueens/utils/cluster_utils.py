def get_cluster_job_id(scheduler_type, output_str_cluster, pbs_scheduler_types):
    """Helper function to retrieve job_id information after submitting a job to
    the job scheduling software.

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

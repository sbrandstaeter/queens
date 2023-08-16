"""Schedulers.

The scheduler package contains a set of scheduler classes which submit
compute jobs either through a job-scheduling software or through a
system call.
"""

VALID_TYPES = {
    'local': ['pqueens.schedulers.local_scheduler', 'LocalScheduler'],
    'cluster': ['pqueens.schedulers.cluster_scheduler', 'ClusterScheduler'],
}

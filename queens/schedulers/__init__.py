"""Schedulers.

The scheduler package contains a set of scheduler classes which submit
compute jobs either through a job-scheduling software or through a
system call.
"""

from queens.schedulers.cluster_scheduler import ClusterScheduler
from queens.schedulers.local_scheduler import LocalScheduler

VALID_TYPES = {
    'local': LocalScheduler,
    'cluster': ClusterScheduler,
}


from schedulers.local_scheduler import LocalScheduler

class SchedulerFactory(object):
    """ Create new scheduler """

    def create_scheduler(scheduler_type):
        """ Create scheduler based on passed scheduler type

        Args:
            scheduler_type (string): what kind of scheduler to create_scheduler
        Returns:
            scheduler: scheduler object
        """

        if scheduler_type == 'local':
            scheduler = LocalScheduler()
        elif scheduler_type == 'PBS':
            raise NotImplementedError
        elif scheduler_type == 'SLURM':
            raise NotImplementedError
        else:
            raise ValueError('Unkown type of scheduler')

        return scheduler

    factory = staticmethod(create_scheduler)

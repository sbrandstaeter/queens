from abc import ABCMeta, abstractmethod

class AbstractScheduler(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def submit(self, job_id, experiment_name, experiment_dir, database_address):
        pass

    @abstractmethod
    def alive(self,process_id):
        pass

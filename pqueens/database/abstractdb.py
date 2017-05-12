from abc import ABCMeta, abstractmethod

class AbstractDB(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def save(self, collection_name):
        pass

    @abstractmethod
    def load(self, collection_name, expt_id):
        pass

from abc import ABCMeta, abstractmethod

class AbstractDesigner(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def sample_generator(self):
        pass

    @abstractmethod
    def get_all_samples(self):
        pass

from abc import ABCMeta, abstractmethod

class AbstractDesigner(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def suggest_next_evaluation(self,params,seed,num_samples):
        pass

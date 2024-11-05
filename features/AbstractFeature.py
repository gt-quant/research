from abc import ABC, abstractmethod

class AbstractFeature(ABC):

    # if you want parameters, put them in the __init__ method

    @abstractmethod
    def get_parents(self):
        pass

    @abstractmethod
    def get_feature(self, df):
        pass

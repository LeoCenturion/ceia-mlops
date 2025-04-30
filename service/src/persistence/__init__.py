from abc import ABC, abstractmethod

class ModelRepository(ABC):
    @abstractmethod
    def load(self, name):
        pass

    @abstractmethod
    def save(self, model):
        pass

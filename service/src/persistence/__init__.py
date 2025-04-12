from abc import ABC, abstractmethod

class ModelRepository(ABC):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self, model):
        pass

from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def train(self, query: str):
        """
        Train the model using data retrieved via the provided query.
        """
        pass

    @abstractmethod
    def predict(self, query: str):
        """
        Predict outcomes for games retrieved using the query.
        """
        pass

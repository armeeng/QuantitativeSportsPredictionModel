from BaseModel import BaseModel

class MLModel(BaseModel):
    def __init__(self, model_name: str, model_type: str):
        super().__init__(model_name)
        self.model_type = model_type  # e.g., 'linear_regression', 'random_forest', etc...

    def train(self, query: str):
        """
        Train using pregame_data from query. Save model to model/modelname + query
        """
        pass

    def predict(self, query: str):
        """
        Load model and predict outcomes on the given query.
        """
        pass

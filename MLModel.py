from BaseModel import BaseModel

class MLModel(BaseModel):
    def __init__(self, model_name: str, model_type: str):
        super().__init__(model_name)
        self.model_type = model_type  # e.g., 'linear_regression', 'random_forest'
        self.model = None  # Placeholder for actual sklearn/torch model instance

    def train(self, query: str, model_save_path: str = None, **kwargs):
        """
        Train using pregame_data from query.
        kwargs may include:
            - subset filters
            - preprocessing flags
        """
        pass

    def predict(self, query: str, model_load_path: str = None, **kwargs):
        """
        Load model and predict outcomes on the given query.
        kwargs may include:
            - output_file_path
        """
        pass

from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def train(self, query: str, **kwargs):
        """
        Train the model using data retrieved via the provided query.
        Optional kwargs can include:
            - model_save_path
            - subset filters
            - additional config
        """
        pass

    @abstractmethod
    def predict(self, query: str, **kwargs):
        """
        Predict outcomes for games retrieved using the query.
        Optional kwargs can include:
            - weights
            - custom indices
            - file paths
            - output format
        """
        pass

    def test(self, query: str):
        """
        Test the model against actual outcomes and betting lines.

        Parameters:
            query (str): SQL query to select games to test on.

        Returns:
            Dict: {
                'model_performance': ...,    # e.g., MSE, MAE
                'betting_performance': ...    # e.g., betting ROI, accuracy
            }
        """

        # 2. Obtain model predictions for the selected games
        predictions = self.predict(query=query)

        # 3. Extract betting lines and actual scores

        # 4. Compare model predictions to actual scores to compute model_performance

        # 5. Compare betting lines to actual scores to compute betting_performance

        # 6. Return performance summary
        return {
            'model_performance': None,   # placeholder for computed metrics
            'betting_performance': None  # placeholder for computed metrics
        }

from BaseModel import BaseModel

class SimilarityModel(BaseModel):
    def __init__(self, model_name: str, distance_metric: str):
        super().__init__(model_name)
        self.distance_metric = distance_metric  # 'cosine', 'euclidean', etc.

    def predict(self,
                query: str,
                reference_query: str = None,
                top_n: int = None,
                custom_indices: list[int] = None,
                use_random_weights: bool = False,
                random_weights: list[float] = None,
                use_internal_weights: bool = False,
                **kwargs):
        """
        Predict using similarity to past games.

        Parameters:
            - query: Games to predict
            - reference_query: Historical games to compare against
            - top_n: Use top N most similar games
            - custom_indices: Override top_n with specific similarity rank indices
            - use_random_weights: Apply provided random_weights
            - use_internal_weights: More similar games carry more weight
        """
        pass

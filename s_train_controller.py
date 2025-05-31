from SimilarityModel import SimilarityModel

# predict next NBA games using 5 nearest neighbors by Euclidean distance,
# weighting by similarity (inverse distance)
sm = SimilarityModel("nba_sim", distance_metric="cosine")
train = sm.train(
    query="SELECT * FROM games WHERE sport='CBB';",
    reference_query="SELECT * FROM games WHERE sport='CBB';",
    top_n=10,
    use_internal_weights=False
)
print(train)
from SimilarityModel import SimilarityModel

# predict next NBA games using 5 nearest neighbors by Euclidean distance,
# weighting by similarity (inverse distance)
sm = SimilarityModel("nba_sim", distance_metric="yule")
train = sm.train(
    query="SELECT * FROM games WHERE sport='NBA';",
    reference_query="SELECT * FROM games WHERE sport='NBA';",
    top_n=5,
    use_internal_weights=True
)
print(train)
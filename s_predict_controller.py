from SimilarityModel import SimilarityModel

# predict next NBA games using 5 nearest neighbors by Euclidean distance,
# weighting by similarity (inverse distance)
sm = SimilarityModel("nba_sim", distance_metric="euclidean")
preds = sm.predict(
    query="SELECT * FROM games WHERE sport='NBA' AND date='2025-05-26';",
    reference_query="SELECT * FROM games WHERE sport='NBA';",
    top_n=100,
    use_internal_weights=True
)
for p in preds:
    print(p)
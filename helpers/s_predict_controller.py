from SimilarityModel import SimilarityModel

# predict next NBA games using 5 nearest neighbors by Euclidean distance,
# weighting by similarity (inverse distance)
sm = SimilarityModel("nba_sim", distance_metric="cosine")
preds = sm.predict(
    query="SELECT * FROM games WHERE sport='NBA' AND date='2025-05-27';",
    reference_query="SELECT * FROM games WHERE sport='NBA';",
    top_n=10,
    use_internal_weights=False
)
for p in preds:
    print(p)
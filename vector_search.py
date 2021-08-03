def kNN_search(query, database, k=10):
    distance_matrix = query.mm(database.extracted_features.t())
    top_scores, most_similar_indices = distance_matrix.topk(k)
    return top_scores, most_similar_indices

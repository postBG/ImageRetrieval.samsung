import torch
from tqdm.notebook import tqdm

from vector_search import kNN_search


@torch.no_grad()
def run_all_retrieval(feature_extractor, valloader, database, k=10):
    all_query_labels = []
    all_retrieved_indices = []
    all_retrieved_labels = []

    feature_extractor.eval()
    for xs, ys, idxs in tqdm(valloader):
        query_features = feature_extractor(xs)
        most_similar_indices = kNN_search(query_features, database, k=k)[1]

        all_query_labels.extend(ys)
        all_retrieved_indices.extend(most_similar_indices)
        all_retrieved_labels.extend(database.targets[most_similar_indices])

    return torch.stack(all_query_labels), torch.stack(all_retrieved_indices), torch.stack(all_retrieved_labels)

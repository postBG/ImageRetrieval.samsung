import numpy as np


def average_precision(label: int, retrieved_labels: np.array, eps=1e-12):
    corrects = np.where(retrieved_labels == label, 1, 0)
    cum_corrects = np.cumsum(corrects)
    denominators = np.arange(1, len(corrects) + 1)
    precisions = cum_corrects / denominators
    average_prevision = np.sum(precisions * corrects) / (np.sum(corrects) + eps)
    return average_prevision


def mean_average_precision(labels, retrieved_labels):
    labels, retrieved_labels = labels.cpu().numpy(), retrieved_labels.cpu().numpy()
    aps = np.array([average_precision(label, retrieved_ls) for label, retrieved_ls in zip(labels, retrieved_labels)])
    return np.mean(aps)

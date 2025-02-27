import numpy as np
import pandas as pd


def recall_at_k(actual, predicted, k):
    actual_set = set(actual)
    predicted_at_k = set(predicted[:k])
    return len(actual_set & predicted_at_k) / k


# Precision values are same as recall in this experiment environment
# def precision_at_k(actual, predicted, k):
#     actual_set = set(actual)
#     predicted_at_k = set(predicted[:k])
#     return len(actual_set & predicted_at_k) / len(predicted_at_k)


def ndcg_at_k(actual, predicted, k):
    actual_set = set(actual)
    predicted_at_k = predicted[:k]
    dcg = sum(
        [1 / np.log2(i + 2) for i, p in enumerate(predicted_at_k) if p in actual_set]
    )
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(actual_set), k))])
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(actual, predicted, k):
    actual_set = set(actual)
    predicted_at_k = predicted[:k]
    score = 0.0
    hit_count = 0.0
    for i, item in enumerate(predicted_at_k):
        if item in actual_set:
            hit_count += 1.0
            score += hit_count / (i + 1)
    return score / len(actual_set) if actual else 0.0

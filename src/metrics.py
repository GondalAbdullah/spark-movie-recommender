"""
Ranking metrics for recommender systems:
Precision@K, Recall@K, MAP@K, NDCG@K
"""

import numpy as np


def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    return len(set(recommended_k) & set(relevant)) / float(k)


def recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    if len(relevant) == 0:
        return 0.0
    return len(set(recommended_k) & set(relevant)) / float(len(relevant))


def average_precision_at_k(recommended, relevant, k):
    score = 0.0
    hits = 0.0

    for i, r in enumerate(recommended[:k], start=1):
        if r in relevant:
            hits += 1.0
            score += hits / i

    # Normalize by number of relevant items
    if len(relevant) == 0:
        return 0.0

    return score / float(min(len(relevant), k))


def map_at_k(recommendations_list, relevant_list, k):
    """
    recommendations_list: list of lists (per user)
    relevant_list       : list of lists (per user)
    """
    scores = []
    for recs, rel in zip(recommendations_list, relevant_list):
        scores.append(average_precision_at_k(recs, rel, k))
    return float(np.mean(scores))


def ndcg_at_k(recommended, relevant, k):
    def dcg(items):
        score = 0.0
        for idx, item in enumerate(items, start=1):
            if item in relevant:
                score += 1.0 / np.log2(idx + 1)
        return score

    recommended_k = recommended[:k]
    ideal_k = relevant[:k]  # ideal list (best-case)

    dcg_score = dcg(recommended_k)
    idcg_score = dcg(ideal_k)

    return dcg_score / idcg_score if idcg_score > 0 else 0.0

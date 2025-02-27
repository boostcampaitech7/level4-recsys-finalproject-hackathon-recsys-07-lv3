import math
import torch

def recall(true: list[list], pred: list[list], normalized: bool = False) -> float:
    recall = 0
    valid_cnt = 0
    for p, t in zip(pred, true):
        if len(t) == 0:
            continue
        if normalized:
            recall += len(set(p) & set(t)) / min(len(p), len(t))
        else:
            recall += len(set(p) & set(t)) / len(t)
        valid_cnt += 1
    try:
        recall /= valid_cnt
    except ZeroDivisionError:
        recall = -1
    return recall

def ndcg(true: list[list], pred: list[list], normalized: bool = False) -> float:
    total_ndcg = 0.0
    valid_count = 0
    for p, t in zip(pred, true):
        if len(t) == 0:
            continue
        dcg = 0.0
        for i, item in enumerate(p):
            if item in t:
                dcg += 1 / math.log2(i + 2)

        ideal_length = min(len(p), len(t)) if normalized else len(t)
        idcg = sum(1 / math.log2(i + 2) for i in range(ideal_length))
        ndcg_score = dcg / idcg if idcg > 0 else 0.0
        total_ndcg += ndcg_score
        valid_count += 1
    try:
        avg_ndcg = total_ndcg / valid_count
    except ZeroDivisionError:
        avg_ndcg = -1
    return avg_ndcg

def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

def rmse_loss(true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((true - pred) ** 2))
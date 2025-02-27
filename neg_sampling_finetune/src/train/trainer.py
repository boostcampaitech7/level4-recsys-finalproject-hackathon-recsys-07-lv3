import numpy as np
import pandas as pd
import torch
import bottleneck as bn
from tqdm import tqdm

from src.train.metrics import *


def evaluate(model, users, user_groups, n_items, top_k, user_train_items, user_actual):
    """
    모델의 평가를 위해 전체 아이템에 대한 추천 리스트를 생성하고
    recall@k 등의 평가 지표를 계산하는 함수.

    Args:
      model (nn.Module): 추천 스코어를 계산하는 모델.
      users (iterable): 평가할 전체 유저 ID 리스트.
      user_groups (dict): 1,3,5,10,15-shot 유저 ID 리스트 딕셔너리.
      n_items (int): 전체 아이템의 개수.
      top_k (int): 상위 추천 아이템 수 (예: 10).
      user_train_items (dict): 각 유저마다 학습에 사용된 아이템 인덱스 리스트 – 평가 시 해당 아이템은 마스킹 처리함.
      user_actual (dict): 각 유저의 실제 정답(ground truth) 아이템 리스트.

    Returns:
      dict: 전체 유저와 1-shot, 3-shot, 5-shot 유저에 대한 평균 recall@k, ndcg@k, map@k 값을 포함하는 딕셔너리.
    """
    model.eval()  # 평가 모드로 전환 (dropout, batchnorm 등 고정)
    all_metrics = {"recall": [], "ndcg": [], "map": []}
    metrics_1 = {"recall": [], "ndcg": [], "map": []}
    metrics_3 = {"recall": [], "ndcg": [], "map": []}
    metrics_5 = {"recall": [], "ndcg": [], "map": []}
    metrics_10 = {"recall": [], "ndcg": [], "map": []}
    metrics_15 = {"recall": [], "ndcg": [], "map": []}

    with torch.no_grad():
        for user in tqdm(users, desc="Evaluating", leave=False):
            # 각 유저에 recommend_items 함수 사용 (학습에 사용한 아이템은 마스킹)
            recommendations = recommend_items(
                model, user, n_items, top_k, user_train_items[user]
            )
            # 추천 결과와 실제 정답(dict나 list)에 따라 메트릭 계산
            recall = recall_at_k(user_actual[user], recommendations, top_k)
            ndcg = ndcg_at_k(user_actual[user], recommendations, top_k)
            map = map_at_k(user_actual[user], recommendations, top_k)

            all_metrics["recall"].append(recall)
            all_metrics["ndcg"].append(ndcg)
            all_metrics["map"].append(map)

            if user in user_groups["users_1"]:
                metrics_1["recall"].append(recall)
                metrics_1["ndcg"].append(ndcg)
                metrics_1["map"].append(map)
            elif user in user_groups["users_3"]:
                metrics_3["recall"].append(recall)
                metrics_3["ndcg"].append(ndcg)
                metrics_3["map"].append(map)
            elif user in user_groups["users_5"]:
                metrics_5["recall"].append(recall)
                metrics_5["ndcg"].append(ndcg)
                metrics_5["map"].append(map)
            elif user in user_groups["users_10"]:
                metrics_10["recall"].append(recall)
                metrics_10["ndcg"].append(ndcg)
                metrics_10["map"].append(map)
            elif user in user_groups["users_15"]:
                metrics_15["recall"].append(recall)
                metrics_15["ndcg"].append(ndcg)
                metrics_15["map"].append(map)

    model.train()  # 평가 후 다시 학습 모드로 전환

    def average_metrics(metrics):
        return {k: sum(v) / len(v) for k, v in metrics.items()}

    return {
        "all": average_metrics(all_metrics),
        "1-shot": average_metrics(metrics_1),
        "3-shot": average_metrics(metrics_3),
        "5-shot": average_metrics(metrics_5),
        "10-shot": average_metrics(metrics_10),
        "15-shot": average_metrics(metrics_15),
    }


def recommend_items(model, user_id, num_items, top_k=10, user_train_items=None):
    """
    단일 유저에 대해 전체 아이템에 대한 스코어를 계산한 후,
    이미 학습에 활용된 아이템(train_items)이 있을 경우 이를 마스킹(-무한대로 대체)하고,
    bn.argpartition을 활용해 상위 top_k 아이템을 효율적으로 추출하는 함수.

    args:
        model: user_id와 item_id의 텐서를 입력받아 스코어를 반환하는 추천 모델.
        user_id (int): 추천을 위한 대상 유저 ID.
        num_items (int): 전체 아이템의 개수.
        top_k (int): 추천할 아이템 수.
        train_items (list 또는 np.array, optional): 학습 시 활용된 해당 유저의 아이템 인덱스 리스트.

    return:
        추천 아이템 인덱스 리스트 (정렬되어 있음)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # user_id를 전체 아이템 개수만큼 반복하여 텐서 생성
    user_ids = torch.full((num_items,), user_id, dtype=torch.long).to(device)
    # 모든 아이템의 인덱스 생성
    item_ids = torch.arange(num_items, dtype=torch.long).to(device)

    with torch.no_grad():
        scores = model(user_ids, item_ids)

    # train_items가 제공되면 해당 아이템의 스코어를 마스킹 처리 (-무한대값으로 대체)
    if user_train_items is not None:
        # torch indexing은 list나 array로도 발동됨
        scores[user_train_items] = -float("inf")

    # GPU에 있을 경우 CPU로 옮긴 후 numpy 배열로 변환
    scores_np = scores.cpu().numpy()

    # bottleneck의 argpartition을 사용하여 상위 top_k의 후보 인덱스를 추출
    # 음수 부호를 취해 내림차순 정렬 효과를 냄.
    candidate_indices = bn.argpartition(-scores_np, top_k - 1)[:top_k]

    # argpartition은 정렬되어 있지 않으므로, 위 후보들에 대해 추가 정렬(내림차순) 수행
    sorted_top_indices = candidate_indices[np.argsort(-scores_np[candidate_indices])]

    return sorted_top_indices.tolist()


def print_metrics(mode, top_k, metrics):
    """
    주어진 메트릭을 형식에 맞춰 출력하는 함수.

    Args:
    - mode (str): 'val', 'best val', 'test' 모드.
    - top_k (int): 상위 추천 아이템 수 (예: 10).
    - metrics (dict): 평가 메트릭 딕셔너리.
    """
    print(
        f"{mode.capitalize()} - All: Recall@{top_k}: {metrics['all']['recall']:.4f}, "
        f"NDCG@{top_k}: {metrics['all']['ndcg']:.4f}, MAP@{top_k}: {metrics['all']['map']:.4f}"
    )

    print(
        f"{mode.capitalize()} - 1-shot: Recall@{top_k}: {metrics['1-shot']['recall']:.4f}, "
        f"NDCG@{top_k}: {metrics['1-shot']['ndcg']:.4f}, MAP@{top_k}: {metrics['1-shot']['map']:.4f}"
    )

    print(
        f"{mode.capitalize()} - 3-shot: Recall@{top_k}: {metrics['3-shot']['recall']:.4f}, "
        f"NDCG@{top_k}: {metrics['3-shot']['ndcg']:.4f}, MAP@{top_k}: {metrics['3-shot']['map']:.4f}"
    )

    print(
        f"{mode.capitalize()} - 5-shot: Recall@{top_k}: {metrics['5-shot']['recall']:.4f}, "
        f"NDCG@{top_k}: {metrics['5-shot']['ndcg']:.4f}, MAP@{top_k}: {metrics['5-shot']['map']:.4f}"
    )

    print(
        f"{mode.capitalize()} - 10-shot: Recall@{top_k}: {metrics['10-shot']['recall']:.4f}, "
        f"NDCG@{top_k}: {metrics['10-shot']['ndcg']:.4f}, MAP@{top_k}: {metrics['10-shot']['map']:.4f}"
    )

    print(
        f"{mode.capitalize()} - 15-shot: Recall@{top_k}: {metrics['15-shot']['recall']:.4f}, "
        f"NDCG@{top_k}: {metrics['15-shot']['ndcg']:.4f}, MAP@{top_k}: {metrics['15-shot']['map']:.4f}"
    )

import pandas as pd
import numpy as np
import torch


def load_data(path):
    # Load files
    # For creating reindex mapping dicts
    full_df = pd.read_csv(path + "full.csv")
    # Used as training data
    train_df = pd.read_csv(path + "train.csv")

    # Used as masking during Top K recommendation
    val_k = pd.read_csv(path + "val_k.csv")
    test_k = pd.read_csv(path + "test_k.csv")

    # Used as ground truths for Top K recommendation
    val_n = pd.read_csv(path + "val_n.csv")
    test_n = pd.read_csv(path + "test_n.csv")

    # Get reindex mapping dicts
    users = full_df["user_id"].unique()
    items = full_df["item_id"].unique()

    user2idx = {user: i for i, user in enumerate(users)}
    item2idx = {item: i for i, item in enumerate(items)}

    # Reindex dataframes
    reindex_df(train_df, user2idx, item2idx)
    reindex_df(val_k, user2idx, item2idx)
    reindex_df(val_n, user2idx, item2idx)
    reindex_df(test_k, user2idx, item2idx)
    reindex_df(test_n, user2idx, item2idx)

    # Store n_user, n_items
    n_users = len(users)
    n_items = len(items)

    # 유니크한 val, test 유저 아이디 리스트
    val_users = val_k["user_id"].unique()
    test_users = test_k["user_id"].unique()

    # 유저 별 인터랙션 개수
    val_inter_counts = val_k.groupby("user_id")["item_id"].count()
    test_inter_counts = test_k.groupby("user_id")["item_id"].count()

    # 아이템의 인기도 리스트
    pop_values = train_df.groupby("item_id").size().to_numpy()
    epsilon = 1e-8
    pop_values_log = np.log(pop_values + epsilon)
    item_popularity = (pop_values_log - pop_values_log.min()) / (
        pop_values_log.max() - pop_values_log.min()
    )

    # 1~15-shot 유저 아이디 리스트
    val_1_users = val_inter_counts[val_inter_counts == 1].index.tolist()
    val_3_users = val_inter_counts[val_inter_counts == 3].index.tolist()
    val_5_users = val_inter_counts[val_inter_counts == 5].index.tolist()
    val_10_users = val_inter_counts[val_inter_counts == 10].index.tolist()
    val_15_users = val_inter_counts[val_inter_counts == 15].index.tolist()

    test_1_users = test_inter_counts[test_inter_counts == 1].index.tolist()
    test_3_users = test_inter_counts[test_inter_counts == 3].index.tolist()
    test_5_users = test_inter_counts[test_inter_counts == 5].index.tolist()
    test_10_users = test_inter_counts[test_inter_counts == 10].index.tolist()
    test_15_users = test_inter_counts[test_inter_counts == 15].index.tolist()

    # 딕셔너리 형태로 모아서 저장
    val_user_groups = {
        "users_1": val_1_users,
        "users_3": val_3_users,
        "users_5": val_5_users,
        "users_10": val_10_users,
        "users_15": val_15_users,
    }

    test_user_groups = {
        "users_1": test_1_users,
        "users_3": test_3_users,
        "users_5": test_5_users,
        "users_10": test_10_users,
        "users_15": test_15_users,
    }

    # {user_id: [item_ids]} 딕셔너리 생성, train 에서는 neg_sampling에서 마스킹으로, Top K 추천에서는 scores에 대한 마스킹으로 사용됨
    train_user_train_items = {
        k: list(v["item_id"].values) for k, v in train_df.groupby("user_id")
    }
    val_user_train_items = {
        k: list(v["item_id"].values) for k, v in val_k.groupby("user_id")
    }
    test_user_train_items = {
        k: list(v["item_id"].values) for k, v in test_k.groupby("user_id")
    }

    # val, test 유저들에 대한 정답 아이템 리스트 딕셔너리
    val_actual = {k: list(v["item_id"].values) for k, v in val_n.groupby("user_id")}
    test_actual = {k: list(v["item_id"].values) for k, v in test_n.groupby("user_id")}

    # train 유저 상호작용 torch.sparse.coo 형태로 변환
    train_coo = create_coo_matrix(train_df, n_users, n_items)

    train_data = {
        "n_users": n_users,
        "n_items": n_items,
        "user2idx": user2idx,
        "item2idx": item2idx,
        "item_popularity": item_popularity,
        "coo": train_coo,
        "user_train_items": train_user_train_items,
    }

    val_data = {
        "users": val_users,
        "user_groups": val_user_groups,
        "user_train_items": val_user_train_items,
        "actual": val_actual,
    }

    test_data = {
        "users": test_users,
        "user_groups": test_user_groups,
        "user_train_items": test_user_train_items,
        "actual": test_actual,
    }

    del full_df, train_df, val_k, test_k, val_n, test_n
    return train_data, val_data, test_data


def reindex_df(df, user_mapping, item_mapping):
    df["user_id"] = df["user_id"].map(user_mapping)
    df["item_id"] = df["item_id"].map(item_mapping)
    return


def create_coo_matrix(df, n_users, n_items):
    # torch 텐서로 변환
    user_id_tensor = torch.tensor(df["user_id"].values, dtype=torch.long)
    item_id_tensor = torch.tensor(df["item_id"].values, dtype=torch.long)
    label_tensor = torch.ones(len(df), dtype=torch.float32)

    # COO 희소 텐서 생성
    indices = torch.stack([user_id_tensor, item_id_tensor])
    values = label_tensor
    size = (n_users, n_items)  # 전체 유저 x 전체 아이템 크기로 지정

    return torch.sparse_coo_tensor(indices, values, size)

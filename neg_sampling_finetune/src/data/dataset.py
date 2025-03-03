import numpy as np
import torch
import random
from torch.utils.data import Dataset


class BPRDataset(Dataset):
    def __init__(self, coo, user_train_items, item_popularity=None):
        """
        Args:
            user_train_items (dict): 사용자별 상호작용 아이템 딕셔너리.
            num_items (int): 전체 아이템 수.
            item_popularity (list or array): 인덱스에 따른 아이템 인기도 (0~1 범위).
        """
        self.users, self.pos_items = coo._indices()
        self.user_train_items = user_train_items
        self.n_items = coo.shape[1]
        self.n_inter = coo._values().shape[0]
        self.item_popularity = item_popularity

        # 아이템 인기도가 주어졌다면, 전역에서 인기도 상위 30%에 해당하는 아이템 리스트를 생성
        if self.item_popularity is not None:
            # 상위 30%에 해당하는 임계값: 70번째 백분위수 이상
            threshold_value = np.percentile(self.item_popularity, 70)
            # 리스트로 변환해서 random.choice가 빠르게 작동하도록 함
            self.top_popular_items = list(
                set(np.where(np.array(self.item_popularity) >= threshold_value)[0])
            )
        else:
            self.top_popular_items = None

    def __len__(self):
        return self.n_inter

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.pos_items[idx]

        if self.top_popular_items is not None:
            neg_item = self._get_pop_neg_item(user.item())
        else:  # No top_popular_items_list
            neg_item = self._get_neg_item(user.item())

        return user, pos_item, neg_item

    def _get_neg_item(self, user):
        train_items = set(self.user_train_items[user])

        neg_item = torch.randint(0, self.n_items, (1,)).item()
        while neg_item in train_items:
            neg_item = torch.randint(0, self.n_items, (1,)).item()
        return neg_item

    def _get_pop_neg_item(self, user):
        train_items = set(self.user_train_items[user])

        neg_item = random.choice(self.top_popular_items)
        for _ in range(10):  # 11 번까지만 상위 30퍼 내에서 무작위 샘플링 시도
            if neg_item in train_items:
                neg_item = random.choice(self.top_popular_items)
            else:
                break
        if (
            neg_item in train_items
        ):  # 11번 내에 샘플링 실패 시 전체 아이템에서 무작위 샘플링
            while neg_item in train_items:
                neg_item = torch.randint(0, self.n_items, (1,)).item()
        return neg_item

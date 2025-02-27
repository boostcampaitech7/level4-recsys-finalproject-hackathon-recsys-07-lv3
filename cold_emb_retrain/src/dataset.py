from functools import cached_property

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack, hstack, diags

class Dataset:
    def __init__(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        self.train_df = train_df
        self.val_df = val_df
        self._check_integrity()

    def _check_integrity(self) -> None:
        ordinal_series = [
            self.train_df['user_id'],
            self.train_df['item_id'],
        ]
        for series in ordinal_series:
            assert series.min() == 0
            assert series.max() == series.nunique() - 1

    @cached_property
    def user_cnt(self) -> int:
        return self.train_df['user_id'].nunique()

    @cached_property
    def item_cnt(self) -> int:
        return self.train_df['item_id'].nunique()

    @cached_property
    def interaction_cnt(self) -> int:
        return len(self.train_df)

    @cached_property
    def density(self) -> float:
        return self.interaction_cnt / (self.user_cnt * self.item_cnt)

    @cached_property
    def user_item_matrix(self) -> csr_matrix:
        user_ids = self.train_df['user_id']
        item_ids = self.train_df['item_id']
        data = np.ones_like(user_ids)
        user_item_matrix = csr_matrix((data, (user_ids, item_ids)), shape=(self.user_cnt, self.item_cnt))
        return user_item_matrix

    @cached_property
    def extended_adj_matrix(self) -> csr_matrix:
        adj_matrix = self.user_item_matrix
        upper_left_zeros = csr_matrix((self.user_cnt, self.user_cnt))
        upper_part = hstack([upper_left_zeros, adj_matrix])
        lower_right_zeros = csr_matrix((self.item_cnt, self.item_cnt))
        lower_part = hstack([adj_matrix.transpose(), lower_right_zeros])
        extended_adj_matrix = vstack([upper_part, lower_part])
        return extended_adj_matrix

    @cached_property
    def normalized_matrix(self) -> csr_matrix:
        row_sum = np.array(self.extended_adj_matrix.sum(axis=1)).squeeze()
        row_sum[row_sum == 0] = 1.0
        normalizer = diags(row_sum ** -0.5)
        normalized_matrix = normalizer @ self.extended_adj_matrix @ normalizer
        return normalized_matrix
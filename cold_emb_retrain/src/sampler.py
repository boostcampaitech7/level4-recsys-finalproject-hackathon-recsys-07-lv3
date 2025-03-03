from tqdm import tqdm
import numpy as np
import torch
import cppimport

from .dataset import Dataset

class Sampler:

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.neg_item_probs = self.get_uniform_neg_item_probs()

    def get_uniform_neg_item_probs(self) -> np.array:
        adj = self.dataset.user_item_matrix
        probs = []
        for user in tqdm(range(self.dataset.user_cnt)):
            positive_items = adj.indices[adj.indptr[user]: adj.indptr[user + 1]]
            cur_probs = np.ones(self.dataset.item_cnt, dtype=np.float32)
            cur_probs[positive_items] = 0
            cur_probs /= cur_probs.sum()
            probs.append(cur_probs)
        return np.array(probs)

    def get_samples(self) -> torch.Tensor:
        adj = self.dataset.user_item_matrix
        indptr = adj.indptr.astype(np.int32)
        indices = adj.indices.astype(np.int32)
        inter_cnt = adj.getnnz(axis=1)
        avg_inter_cnt = inter_cnt.mean()
        sample_num_per_user = np.full(self.dataset.user_cnt, avg_inter_cnt, dtype=np.int32)

        sampler_cpp = cppimport.imp("src.sampler_cpp")
        samples_np = sampler_cpp.get_samples_cpp(self.dataset.user_cnt, self.dataset.item_cnt, sample_num_per_user,
                                                 self.neg_item_probs, indptr, indices)

        return torch.tensor(samples_np, dtype=torch.long)
import torch

from .dataset import Dataset

class LightGCN(torch.nn.Module):
    def __init__(self, dataset: Dataset, config: dict) -> None:
        super(LightGCN, self).__init__()
        self.dataset = dataset
        self.config = config
        self.user_embeddings = torch.nn.Embedding(
            self.dataset.user_cnt, config['emb_size'], dtype=torch.float32
        )
        self.item_embeddings = torch.nn.Embedding(
            self.dataset.item_cnt, config['emb_size'], dtype=torch.float32
        )
        self.aggregator = self.get_aggregator()
        torch.nn.init.normal_(self.user_embeddings.weight, std=0.1)
        torch.nn.init.normal_(self.item_embeddings.weight, std=0.1)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        user_embeddings, item_embeddings = self.get_embeddings()
        user_embeddings = user_embeddings[user_indices]
        item_embeddings = item_embeddings[item_indices]
        return torch.sum(user_embeddings * item_embeddings, dim=1)

    def get_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = []
        full_embedding = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        embeddings.append(full_embedding)
        for _ in range(self.config['num_layers']):
            full_embedding = torch.sparse.mm(self.aggregator, full_embedding)
            embeddings.append(full_embedding)
        final_embedding = torch.stack(embeddings, dim=0).mean(dim=0)
        final_user_embedding, final_item_embedding = torch.split(
            final_embedding, [self.dataset.user_cnt, self.dataset.item_cnt])
        return final_user_embedding, final_item_embedding

    def get_aggregator(self) -> tuple[torch.Tensor, torch.Tensor]:
        adj = self.dataset.normalized_matrix.tocsr()
        crow_indices = torch.tensor(adj.indptr, dtype=torch.int64)
        col_indices = torch.tensor(adj.indices, dtype=torch.int64)
        values = torch.tensor(adj.data, dtype=torch.float32)
        aggregator = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=adj.shape, requires_grad=False)
        return aggregator

    def get_topk(self, k: int) -> torch.Tensor:
        trues = self.dataset.train_df.groupby('user_id')['item_id'].apply(list)
        user_embeddings, item_embeddings = self.get_embeddings()
        scores = user_embeddings @ item_embeddings.T

        for user_id, items in trues.items():
            scores[user_id, items] = torch.finfo(torch.float32).min
        topk = torch.topk(scores, k=k, dim=1).indices
        return topk

    def to(self, device: torch.device):
        super(LightGCN, self).to(device)
        self.aggregator = self.aggregator.to(device)
        self.user_embeddings = self.user_embeddings.to(device)
        self.item_embeddings = self.item_embeddings.to(device)
        return self
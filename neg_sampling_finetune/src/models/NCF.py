import torch
import torch.nn as nn


class NCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64, dropout=0.2):
        super(NCF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.dropout = dropout

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.mlp = nn.Sequential(  # [batch_size, emb_dim * 2]
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),  # [batch_size, emb_dim]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim // 2),  # [batch_size, emb_dim // 2]
            nn.ReLU(),
            nn.Linear(emb_dim // 2, 1),  # [batch_size, 1]
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, user, item):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        concat = torch.cat([user_emb, item_emb], dim=-1)  # [batch_size, emb_dim * 2]
        return self.mlp(concat).squeeze()  # [batch_size]

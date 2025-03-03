from tqdm import tqdm
import torch

from .model import LightGCN
from .dataset import Dataset
from .sampler import Sampler
from .utils import recall, ndcg, bpr_loss

class Trainer:

    def __init__(self, model: LightGCN, dataset: Dataset, sampler: Sampler, config: dict):
        self.model = model
        self.dataset = dataset
        self.sampler = sampler
        self.config = config

        self.user_group_thresholds = [(1, 1), (3, 3), (5, 5), (0, 10000000)]
        self.user_group_ids = []
        self.user_group_trues = []

        user_value_cnts = self.dataset.train_df['user_id'].value_counts()
        for (lower, upper) in self.user_group_thresholds:
            cur_user_ids = user_value_cnts[(user_value_cnts >= lower) & (user_value_cnts <= upper)].index.tolist()
            self.user_group_ids.append(cur_user_ids)
            print(f'{len(cur_user_ids)} users within range [{lower}, {upper}]')

        grouped = self.dataset.val_df.groupby('user_id')['item_id'].apply(list)
        for cur_user_ids in tqdm(self.user_group_ids):
            cur_trues = [grouped[user_id] if user_id in cur_user_ids else [] for user_id in range(self.dataset.user_cnt)]
            self.user_group_trues.append(cur_trues)

    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        self.model.to(device)

        best_score = 0.0
        best_model_state_dict = None
        early_stop_counter = 0

        for epoch in range(self.config['epochs']):
            pairwise_samples = self.sampler.get_samples()
            pairwise_samples = pairwise_samples.to(device)
            dataset = torch.utils.data.TensorDataset(*pairwise_samples.T)
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=0,
            )
            for users, pos_samples, neg_samples in tqdm(dataloader, desc=f'Epoch {epoch}'):
                pos_scores = self.model(users, pos_samples)
                neg_scores = self.model(users, neg_samples)
                loss = bpr_loss(pos_scores, neg_scores)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
  
            if epoch % self.config['validate_freq'] == 0:
                cur_val_score = self.validate()
                if cur_val_score > best_score:
                    best_score = cur_val_score
                    best_model_state_dict = self.model.state_dict()
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= self.config['patience']:
                        print(f'Early stopping triggered at epoch {epoch}')
                        torch.save(best_model_state_dict, f"saved/{self.config['dataset']}_best_model.pt")
                        return

    def refresh_cold_users(self):
        value_cnts = self.dataset.train_df['user_id'].value_counts()
        cold_user_ids = value_cnts[value_cnts <= self.config['refresh_threshold']].index.tolist()
        with torch.no_grad():
            self.model.user_embeddings.weight[cold_user_ids] = torch.zeros(self.config['emb_size'], dtype=torch.float32)
            torch.nn.init.normal_(self.model.user_embeddings.weight[cold_user_ids], std=0.1)

    def validate(self) -> None:
        self.model.eval()
        with torch.no_grad():
            pred = self.model.get_topk(10).to('cpu').numpy().tolist()
        self.model.train()

        cur_recalls = []
        cur_ndcgs = []
        for cur_user_group_trues in self.user_group_trues:
            cur_recall = recall(cur_user_group_trues, pred)
            cur_recalls.append(cur_recall)
            cur_ndcg = ndcg(cur_user_group_trues, pred)
            cur_ndcgs.append(cur_ndcg)
        print(f"Recall@10 - {' '.join(f'{r:.4f}' for r in cur_recalls)}")
        print(f"NDCG@10   - {' '.join(f'{n:.4f}' for n in cur_ndcgs)}")
        print('')
        return cur_ndcgs[-1]
import numpy as np
import pandas as pd
import torch
import argparse

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from src import Dataset, LightGCN, Sampler, Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--emb_size', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--dataset', type=str, default='ml-1m')
parser.add_argument('--retrain', action='store_true')
parser.add_argument('--refresh_threshold', type=int, default=5)
parser.add_argument('--retrain_epochs', type=int, default=3)

args = parser.parse_args()
config = vars(args)

if config['dataset'] == 'ml-1m':
    train_df = pd.read_csv(f"data/{config['dataset']}/train.csv")
else:
    train1_df = pd.read_csv(f"data/{config['dataset']}/train1.csv")
    train2_df = pd.read_csv(f"data/{config['dataset']}/train2.csv")
    train_df = pd.concat([train1_df, train2_df], ignore_index=True)
val_df = pd.read_csv(f"data/{config['dataset']}/val.csv")
dataset = Dataset(train_df, val_df)
model = LightGCN(dataset, config)  
sampler = Sampler(dataset)

if config['retrain']:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(f"saved/{config['dataset']}_best_model.pt", weights_only=True, map_location=device))
    trainer = Trainer(model, dataset, sampler, config)
    print('Before retraining:')
    trainer.validate()
    trainer.refresh_cold_users()
    print('After retraining:')
    config['epochs'] = config['retrain_epochs']
    trainer.train()
else:
    trainer = Trainer(model, dataset, sampler, config)
    trainer.train()
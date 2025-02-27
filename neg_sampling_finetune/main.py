import os
from functools import partial

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataloader import load_data
from src.data.dataset import BPRDataset
from src.models.NCF import NCF
from src.train.loss import BPRLoss
from src.train.trainer import evaluate, print_metrics, recommend_items
from src.utils.setting import set_seed, worker_init_fn

path = "/data/ephemeral/home/data/cold/"
save_dir = "../saved/"
model_path = save_dir + "ncf.pth"
result_path = save_dir + "result.csv"

# 딕셔너리 형태로 데이터 로드
train_data, val_data, test_data = load_data(path)

SEED = 42
set_seed(SEED)
# worker_init_fn에 SEED를 전달하는 partial 함수 생성, 데이터로더 셔플 재현성을 위함
worker_init_fn_with_seed = partial(worker_init_fn, seed=SEED)


########################################## PRE-TRAINING ##########################################
# Pre-train 데이터셋, 데이터로더
pre_train_dataset = BPRDataset(train_data["coo"], train_data["user_train_items"])
pre_train_loader = DataLoader(
    pre_train_dataset,
    batch_size=1024,
    shuffle=True,
    num_workers=4,
    worker_init_fn=worker_init_fn_with_seed,
)


# 하이퍼 파라미터 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emb_dim = 64
dropout = 0.2
model = NCF(train_data["n_users"], train_data["n_items"], emb_dim, dropout).to(device)
criterion = BPRLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# 학습 파라미터 설정
epochs = 1
valid_interval = 1
early_stop = 10

top_k = 10

epochs_after_best = 0  # 얼리 스탑을 위한 카운터 변수
best_val_ndcg = None  # 얼리 스탑에 비교대상이 되는 ndcg 저장 변수
best_val_metrics = None  # 최고 ndcg 시점의 모든 메트릭을 저장하는 변수

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for user, pos_item, neg_item in tqdm(
        pre_train_loader, desc=f"Training Epoch {epoch+1}"
    ):
        user = user.to(device)
        pos_item = pos_item.to(device)
        neg_item = neg_item.to(device)

        pos_scores = model(user, pos_item)
        neg_scores = model(user, neg_item)

        loss = criterion(pos_scores, neg_scores)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(pre_train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_train_loss:.4f}")

    # 평가: val_users, n_items, top_k, 그리고 유저별 학습/정답 아이템 정보 사용
    if (epoch + 1) % valid_interval == 0:
        # Evaluate on validation users
        val_metrics = evaluate(
            model,
            val_data["users"],
            val_data["user_groups"],
            train_data["n_items"],
            top_k,
            val_data["user_train_items"],
            val_data["actual"],
        )
        print_metrics("val", top_k, val_metrics)

        # Check if this is the best validation NDCG
        if best_val_ndcg is None or val_metrics["all"]["ndcg"] > best_val_ndcg:
            best_val_ndcg = val_metrics["all"]["ndcg"]
            best_val_metrics = val_metrics
            best_epoch = epoch + 1

            # Save model
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Current best epoch:{best_epoch}, Model saved to {model_path}")
            epochs_after_best = 0  # Resetting counter
        else:
            epochs_after_best += 1
        if epochs_after_best >= early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Evaluate on test users
model.load_state_dict(torch.load(model_path, weights_only=True))
test_metrics = evaluate(
    model,
    test_data["users"],
    test_data["user_groups"],
    train_data["n_items"],
    top_k,
    test_data["user_train_items"],
    test_data["actual"],
)

# Print best validation metrics
print(f"Best epoch: {best_epoch}")
print_metrics("best val", top_k, best_val_metrics)
# Print test metrics
print(f"Test results from best epoch {best_epoch}")
print_metrics("test", top_k, test_metrics)


########################################## FINE-TUNE TRAINING ##########################################
finetune_dataset = BPRDataset(
    train_data["coo"], train_data["user_train_items"], train_data["item_popularity"]
)
finetune_loader = DataLoader(
    finetune_dataset,
    batch_size=1024,
    shuffle=True,
    num_workers=4,
    worker_init_fn=worker_init_fn_with_seed,
)

model.load_state_dict(torch.load(model_path, weights_only=True))
criterion = BPRLoss()
optimizer = nn.optim.Adam(model.parameters(), lr=0.0005)


epochs = 1
valid_interval = 1
early_stop = 10

top_k = 10

epochs_after_best = 0  # 얼리 스탑을 위한 카운터 변수
best_val_ndcg = None  # 얼리 스탑에 비교대상이 되는 ndcg 저장 변수
best_val_metrics = None  # 최고 ndcg 시점의 모든 메트릭을 저장하는 변수

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for user, pos_item, neg_item in tqdm(
        finetune_loader, desc=f"Training Epoch {epoch+1}"
    ):
        user = user.to(device)
        pos_item = pos_item.to(device)
        neg_item = neg_item.to(device)

        pos_scores = model(user, pos_item)
        neg_scores = model(user, neg_item)

        loss = criterion(pos_scores, neg_scores)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(finetune_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_train_loss:.4f}")

    # 평가: val_users, n_items, top_k, 그리고 유저별 학습/정답 아이템 정보 사용
    if (epoch + 1) % valid_interval == 0:
        # Evaluate on validation users
        val_metrics = evaluate(
            model,
            val_data["users"],
            val_data["user_groups"],
            train_data["n_items"],
            top_k,
            val_data["user_train_items"],
            val_data["actual"],
        )
        print_metrics("val", top_k, val_metrics)

        # Check if this is the best validation NDCG
        if best_val_ndcg is None or val_metrics["all"]["ndcg"] > best_val_ndcg:
            best_val_ndcg = val_metrics["all"]["ndcg"]
            best_val_metrics = val_metrics
            best_epoch = epoch + 1

            # Save model
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Current best epoch:{best_epoch}, Model saved to {model_path}")
            epochs_after_best = 0  # Resetting counter
        else:
            epochs_after_best += 1
        if epochs_after_best >= early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Evaluate on test users
model.load_state_dict(torch.load(model_path, weights_only=True))
test_metrics = evaluate(
    model,
    test_data["users"],
    test_data["user_groups"],
    train_data["n_items"],
    top_k,
    test_data["user_train_items"],
    test_data["actual"],
)

# Print best validation metrics
print(f"Best epoch: {best_epoch}")
print_metrics("best val", top_k, best_val_metrics)
# Print test metrics
print(f"Test results from best epoch {best_epoch}")
print_metrics("test", top_k, test_metrics)


model.eval()
recommendations = []

# 각 사용자에 대해 추천 아이템을 생성하고 리스트에 추가
with torch.no_grad():
    for user in tqdm(
        test_data["users"],
        desc="Generating recommendation list for test users",
        leave=False,
    ):
        recommended_items = recommend_items(
            model, user, train_data["n_items"], 10, test_data["user_train_items"][user]
        )
        for item in recommended_items:
            recommendations.append({"user_id": user, "item_id": item})

# 리스트를 데이터프레임으로 변환
result = pd.DataFrame(recommendations)

idx2user = {v: k for k, v in train_data["user2idx"].items()}
idx2item = {v: k for k, v in train_data["item2idx"].items()}

result["user_id"] = result["user_id"].map(idx2user)
result["item_id"] = result["item_id"].map(idx2item)

result.to_csv(result_path, index=False)
print(f"Recommendation list saved at {result_path}")

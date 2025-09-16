# src/train.py

import copy
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from src.config import Config
# from src.utils import get_metrics  # optionally move metric logic here later


class Mapper(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.reset()

    def reset(self):
        with torch.no_grad():
            self.linear.weight.data = torch.eye(self.linear.out_features).to(Config.DEVICE)
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor):
        return self.linear(x)
def get_positive_indices(dataset, query_idx, retrieved, total, mode):
    positives = list(dataset.relation_list[query_idx])
    if mode == "random":
        return random.sample(positives, min(total, len(positives)))
    
    if "worst" in mode:
        retrieved = np.flipud(retrieved)
    
    pos = [i for i in retrieved if i in positives]
    return pos[:total]


def get_negative_indices(dataset, query_idx, retrieved, total, mode):
    positives = set(dataset.relation_list[query_idx])
    if mode == "random":
        candidates = list(dataset.train_set - positives)
        return random.sample(candidates, total)

    if "worst" in mode:
        retrieved = np.flipud(retrieved)

    neg = [i for i in retrieved if i not in positives]
    return neg[:total]


def get_targets(dataset, passage_embs, baseline_retrieved, query_indices, total, pos_ratio, mode):
    pos_embs = []
    neg_embs = []

    for idx in query_indices:
        pos_count = max(1, round(total * pos_ratio))
        neg_count = total - pos_count
        retrieved = baseline_retrieved[idx]

        pos_idx = get_positive_indices(dataset, idx, retrieved, pos_count, mode)
        neg_idx = get_negative_indices(dataset, idx, retrieved, neg_count, mode)

        pos_emb = passage_embs[pos_idx] if pos_idx else np.empty((0, passage_embs.shape[1]))
        neg_emb = passage_embs[neg_idx] if neg_idx else np.empty((0, passage_embs.shape[1]))

        pos_embs.append(torch.tensor(pos_emb, device=Config.DEVICE, dtype=torch.float))
        neg_embs.append(torch.tensor(neg_emb, device=Config.DEVICE, dtype=torch.float))

    return pos_embs, neg_embs
def compute_loss(mapped_queries, positives, negatives, margin=0.2, norm=2):
    batch_size = len(mapped_queries)
    losses = []

    for i in range(batch_size):
        pos = torch.nanmean(positives[i], dim=0)
        neg = torch.nanmean(negatives[i], dim=0)
        query = mapped_queries[i]

        d_pos = torch.norm(query - pos, p=norm)
        d_neg = torch.norm(query - neg, p=norm)

        loss = d_pos ** 2 + torch.relu(margin - d_neg) ** 2
        losses.append(loss)

    return torch.stack(losses).mean()


def train(mapper, dataset, query_embs, passage_embs, baseline_indices,
          epochs=50, patience=3, batch_size=512, lr=1e-3,
          total=4, pos_ratio=0.75, mode="worst-worst", margin=0.2, norm=2):

    optimizer = optim.Adam(mapper.parameters(), lr=lr)
    best_mapper = copy.deepcopy(mapper)
    best_score = -float("inf")
    no_improve = 0

    for epoch in range(epochs):
        mapper.train()
        losses = []

        with tqdm(total=len(dataset.train_set) // batch_size, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for _ in range(len(dataset.train_set) // batch_size):
                query_idx = random.sample(list(dataset.train_set), batch_size)
                q_emb = torch.tensor(query_embs[query_idx], device=Config.DEVICE, dtype=torch.float)
                mapped_q = mapper(q_emb)

                pos, neg = get_targets(dataset, passage_embs, baseline_indices, query_idx, total, pos_ratio, mode)
                loss = compute_loss(mapped_q, pos, neg, margin=margin, norm=norm)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                pbar.set_postfix({"loss": np.mean(losses)})
                pbar.update(1)

        # Evaluate on validation
        mapper.eval()
        with torch.no_grad():
            val_idx = list(dataset.validation_set)
            val_emb = torch.tensor(query_embs[val_idx], device=Config.DEVICE, dtype=torch.float)
            mapped_val = mapper(val_emb).cpu().numpy()

            val_retrieved = dataset.search(mapped_val)
            _, _, _, pessimistic_mrr = dataset.get_metrics(val_idx, val_retrieved)

        if pessimistic_mrr > best_score:
            best_score = pessimistic_mrr
            best_mapper = copy.deepcopy(mapper)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return best_mapper

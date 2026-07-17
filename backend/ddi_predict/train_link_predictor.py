"""
Step 3 of the DDI prediction pipeline: train a GCN link-prediction model on
the DDInter interaction graph, using OpenAI SMILES embeddings (from
embed_smiles.py) as node features.

Mirrors the approach in DDI-LLM-main/PyG__link_pred_DDI.ipynb: a 2-layer
GCN encoder produces node embeddings, and a dot-product decoder scores
edge likelihood. Trained with random negative sampling on held-out edges.

Usage:
    python train_link_predictor.py [--epochs 100] [--hidden 128] [--out-dim 64] [--lr 0.01]
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling

from common import GCNEncoder, INTERACTIONS_CSV, MODEL_PATH, decode, load_edges, load_embeddings, standardize_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--out-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("[info] Loading embeddings...")
    emb_by_num = load_embeddings()
    nums = sorted(emb_by_num.keys())
    node_index = {num: i for i, num in enumerate(nums)}
    x = torch.tensor(np.stack([emb_by_num[n] for n in nums]), dtype=torch.float)
    x, feat_mean, feat_std = standardize_features(x)
    print(f"[info] {len(nums):,} nodes, {x.shape[1]}-dim features (standardized)")

    print(f"[info] Loading edges from {INTERACTIONS_CSV}")
    edges = load_edges(node_index)
    if len(edges) < 10:
        raise SystemExit(f"Only {len(edges)} known edges among embedded drugs — not enough to train on.")
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # undirected
    print(f"[info] {len(edges):,} unique interaction edges among embedded drugs")

    data = Data(x=x, edge_index=edge_index)
    transform = RandomLinkSplit(
        num_val=0.05, num_test=0.1, is_undirected=True,
        add_negative_train_samples=False, split_labels=True,
    )
    train_data, val_data, test_data = transform(data)

    model = GCNEncoder(x.shape[1], args.hidden, args.out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def train_epoch() -> float:
        model.train()
        optimizer.zero_grad()
        z = model(train_data.x, train_data.edge_index)

        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index,
            num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.pos_edge_label_index.size(1),
            method="sparse",
        )
        edge_label_index = torch.cat([train_data.pos_edge_label_index, neg_edge_index], dim=-1)
        edge_label = torch.cat([
            torch.ones(train_data.pos_edge_label_index.size(1)),
            torch.zeros(neg_edge_index.size(1)),
        ], dim=0)

        out = decode(z, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(out, edge_label)
        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def evaluate(split_data) -> tuple[float, float]:
        model.eval()
        z = model(train_data.x, train_data.edge_index)  # message-pass over train graph only
        pos = torch.sigmoid(decode(z, split_data.pos_edge_label_index))
        neg = torch.sigmoid(decode(z, split_data.neg_edge_label_index))
        scores = torch.cat([pos, neg]).numpy()
        labels = torch.cat([torch.ones(pos.size(0)), torch.zeros(neg.size(0))]).numpy()
        return roc_auc_score(labels, scores), average_precision_score(labels, scores)

    print(f"[info] Training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch()
        if epoch % 10 == 0 or epoch == args.epochs:
            val_auc, val_ap = evaluate(val_data)
            print(f"[info] epoch {epoch:3d}  loss={loss:.4f}  val_auc={val_auc:.4f}  val_ap={val_ap:.4f}")

    test_auc, test_ap = evaluate(test_data)
    print(f"[info] Final test AUROC={test_auc:.4f}  AUPR={test_ap:.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": x.shape[1],
        "hidden_dim": args.hidden,
        "out_dim": args.out_dim,
        "node_order": nums,  # index -> ddinter_num, needed by predict_new_interactions.py
        "feat_mean": feat_mean,  # replay the same feature standardization at inference
        "feat_std": feat_std,
        "test_auc": test_auc,
        "test_ap": test_ap,
    }, MODEL_PATH)
    print(f"[info] Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()

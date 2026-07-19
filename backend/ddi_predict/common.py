"""
Shared graph/model utilities for train_link_predictor.py and
predict_new_interactions.py (steps 3 and 4 of the DDI prediction pipeline).
Kept separate from fetch_smiles.py/embed_smiles.py so those lighter-weight
steps don't require torch/torch_geometric to be installed.
"""
import csv
import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

HERE = Path(__file__).resolve().parent
BACKEND_DIR = HERE.parent
INTERACTIONS_CSV = BACKEND_DIR / "interactions.csv"
SMILES_CSV = HERE / "smiles_cache.csv"
EMBEDDINGS_NPZ = HERE / "embeddings.npz"
MODEL_PATH = HERE / "model.pt"


def load_embeddings() -> dict[int, np.ndarray]:
    if not EMBEDDINGS_NPZ.exists():
        raise SystemExit(f"No embeddings at {EMBEDDINGS_NPZ} — run embed_smiles.py first.")
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    nums = data["ddinter_nums"]
    vecs = data["embeddings"]
    return {int(n): v for n, v in zip(nums, vecs)}


def load_drug_names() -> dict[int, str]:
    """{ddinter_num: name}, sourced from smiles_cache.csv (built in step 1)."""
    names: dict[int, str] = {}
    with open(SMILES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names[int(row["ddinter_num"])] = row["ddinter_name"]
    return names


def _is_data_row(row: list[str]) -> bool:
    try:
        float(row[4])
        return True
    except (ValueError, IndexError):
        return False


def load_edges(node_index: dict[int, int]) -> set[tuple[int, int]]:
    """
    Known interaction edges among embedded drugs, as (i, j) node-index pairs
    with i < j (deduplicated, undirected).
    """
    edges: set[tuple[int, int]] = set()
    with open(INTERACTIONS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first and not _is_data_row(first):
            rows = reader
        else:
            rows = itertools.chain([first], reader) if first else reader
        for row in rows:
            if len(row) < 4:
                continue
            try:
                a = int(row[0].strip()[7:])
                b = int(row[2].strip()[7:])
            except (ValueError, IndexError):
                continue
            if a in node_index and b in node_index and a != b:
                i, j = node_index[a], node_index[b]
                edges.add((min(i, j), max(i, j)))
    return edges


class GCNEncoder(torch.nn.Module):
    """
    2-layer GCN with a linear residual from the raw input features straight
    to the output. The DDInter graph is very dense (avg degree ~140 among
    ~900 embedded drugs), so plain stacked GCNConv layers oversmooth badly —
    every node's neighborhood reaches most of the graph within 2 hops,
    collapsing all node embeddings to nearly the same vector. The skip
    connection guarantees per-node identity (from the SMILES embedding)
    always survives in z, regardless of how much the conv branch smooths.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.skip = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h + self.skip(x)


def decode(z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Dot-product edge decoder: returns raw logits (apply sigmoid for probability)."""
    return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)


def standardize_features(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Zero-mean/unit-variance per feature dimension. Raw transformer hidden
    states fed straight into a freshly-initialized GCN tend to produce huge
    early-training loss spikes and slow/unstable convergence — standardizing
    first keeps gradients well-scaled. Returns (standardized_x, mean, std)
    so the same transform can be replayed at inference time.
    """
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (x - mean) / std, mean, std

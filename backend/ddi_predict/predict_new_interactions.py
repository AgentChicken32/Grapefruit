"""
Step 4 of the DDI prediction pipeline: score every currently-unconnected
drug pair with the trained GCN link predictor and write high-confidence
candidates to ../predicted_interactions.csv.

These are AI-generated, NOT clinically verified. main.py loads them into a
separate `predicted_interactions` table and the UI marks them with an
advisory badge distinct from verified DDInter interactions.

Usage:
    python predict_new_interactions.py [--threshold 0.90] [--max-per-drug 20]
"""
import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import torch

from common import HERE, BACKEND_DIR, GCNEncoder, MODEL_PATH, decode, load_drug_names, load_edges, load_embeddings

OUTPUT_CSV = BACKEND_DIR / "predicted_interactions.csv"
MANIFEST_JSON = HERE / "embeddings_manifest.json"


def method_label() -> str:
    if MANIFEST_JSON.exists():
        manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
        return f"gcn-{manifest.get('backend', 'unknown')}-{manifest.get('model', 'unknown')}-v1"
    return "gcn-smiles-v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.90,
                         help="Minimum predicted probability to keep a candidate pair")
    parser.add_argument("--max-per-drug", type=int, default=20,
                         help="Cap on kept candidates per drug, to keep the list reviewable")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"No trained model at {MODEL_PATH} — run train_link_predictor.py first.")

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    nums: list[int] = ckpt["node_order"]
    node_index = {num: i for i, num in enumerate(nums)}
    n = len(nums)

    emb_by_num = load_embeddings()
    x = torch.tensor(np.stack([emb_by_num[num] for num in nums]), dtype=torch.float)
    if "feat_mean" in ckpt:
        x = (x - ckpt["feat_mean"]) / ckpt["feat_std"]

    edges = load_edges(node_index)
    edge_pairs = set(edges)  # (i, j) with i < j, already-known interactions
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    model = GCNEncoder(ckpt["in_dim"], ckpt["hidden_dim"], ckpt["out_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        z = model(x, edge_index)

    print(f"[info] Scoring candidate pairs among {n:,} drugs (~{n * (n - 1) // 2:,} possible pairs)...")

    candidates: list[tuple[float, int, int]] = []
    with torch.no_grad():
        for i in range(n):
            js = torch.arange(i + 1, n)
            if len(js) == 0:
                continue
            pair_index = torch.stack([torch.full_like(js, i), js])
            scores = torch.sigmoid(decode(z, pair_index))
            for j, score in zip(js.tolist(), scores.tolist()):
                if (i, j) in edge_pairs:
                    continue
                if score >= args.threshold:
                    candidates.append((score, i, j))
            if i % 200 == 0:
                print(f"[info] {i:,}/{n:,} drugs scored, {len(candidates):,} candidates so far")

    print(f"[info] {len(candidates):,} pairs at/above threshold {args.threshold}")

    candidates.sort(key=lambda c: c[0], reverse=True)
    per_drug_count: dict[int, int] = defaultdict(int)
    kept: list[tuple[float, int, int]] = []
    for score, i, j in candidates:
        if per_drug_count[i] >= args.max_per_drug or per_drug_count[j] >= args.max_per_drug:
            continue
        kept.append((score, i, j))
        per_drug_count[i] += 1
        per_drug_count[j] += 1

    print(f"[info] {len(kept):,} pairs kept after per-drug cap ({args.max_per_drug})")

    names = load_drug_names()
    generated_at = datetime.now(timezone.utc).isoformat()
    method = method_label()

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["drug_a_id", "drug_a_name", "drug_b_id", "drug_b_name", "confidence", "method", "generated_at"])
        for score, i, j in kept:
            num_a, num_b = nums[i], nums[j]
            writer.writerow([
                f"DDInter{num_a}", names.get(num_a, ""),
                f"DDInter{num_b}", names.get(num_b, ""),
                round(score, 4), method, generated_at,
            ])

    print(f"[info] Wrote {len(kept):,} predicted interactions to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

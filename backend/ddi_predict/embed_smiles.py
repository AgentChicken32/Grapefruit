"""
Step 2 of the DDI prediction pipeline: embed every cached drug's chemical
structure AND textual metadata (name, ATC category, IUPAC name, PubChem
description — see fetch_smiles.py). These embeddings become the node
features for the link-prediction GCN trained in train_link_predictor.py.

Backends:
  openai  Requires OPENAI_API_KEY. One combined string per drug (name +
          ATC + SMILES + IUPAC + description) is embedded in a single
          call — OpenAI's embedding model handles SMILES and natural
          language fine within the same text. Costs are small but the
          account needs active billing/quota.

  local   Free, offline. Two separate models, concatenated per drug —
          this mirrors DDI-LLM's own "multi-view" approach of treating
          structure and description as distinct feature views rather
          than one blob of mixed syntax:
            - SMILES  -> ChemBERTa (seyonec/ChemBERTa-zinc-base-v1),
                         pretrained specifically on SMILES tokens.
            - name + ATC category + IUPAC name + description ->
                         a general text model (bert-base-uncased by
                         default), mean-pooled.
          Feeding natural-language text into ChemBERTa's SMILES-only
          tokenizer would mostly produce [UNK] tokens, so the two views
          are embedded separately and concatenated rather than merged
          into one string.

  SMILES is optional: a drug with no resolvable PubChem structure still
  gets a feature vector, using a zero-filled structure half and real
  name/ATC-category text for the other half — so drugs outside the
  ~907 with PubChem CIDs aren't excluded from the prediction graph
  entirely, they just carry weaker (text-only) signal.

Usage:
    python embed_smiles.py --backend openai [--model text-embedding-3-small] [--batch-size 100]
    python embed_smiles.py --backend local  [--model seyonec/ChemBERTa-zinc-base-v1] \\
                            [--text-model bert-base-uncased] [--batch-size 32]
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SMILES_CSV = HERE / "smiles_cache.csv"
EMBEDDINGS_NPZ = HERE / "embeddings.npz"
MANIFEST_JSON = HERE / "embeddings_manifest.json"

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_SMILES_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
DEFAULT_TEXT_MODEL = "bert-base-uncased"


def load_drug_data(path: Path) -> dict[int, dict]:
    """
    {ddinter_num: {name, atc_category, smiles, iupac_name, description}}

    SMILES is optional — a drug with no resolvable PubChem structure still
    gets a row (name + ATC category, when known) so it isn't dropped from
    the prediction graph entirely; it just falls back to text-only signal.
    Only a drug with no name at all (shouldn't happen) is skipped.
    """
    rows: dict[int, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = (row.get("ddinter_name") or "").strip()
            if not name:
                continue
            rows[int(row["ddinter_num"])] = {
                "name": name,
                "atc_category": (row.get("atc_category") or "").strip(),
                "smiles": (row.get("smiles") or "").strip(),
                "iupac_name": (row.get("iupac_name") or "").strip(),
                "description": (row.get("description") or "").strip(),
            }
    return rows


def info_text(drug: dict) -> str:
    """Natural-language parts only — no SMILES (that's ChemBERTa's job)."""
    parts = [drug["name"]]
    if drug["atc_category"]:
        parts.append(f"ATC category: {drug['atc_category']}")
    if drug["iupac_name"]:
        parts.append(f"IUPAC name: {drug['iupac_name']}")
    if drug["description"]:
        parts.append(f"Description: {drug['description']}")
    return ". ".join(parts)


def combined_text(drug: dict) -> str:
    """Everything in one string — used for the OpenAI backend."""
    parts = [drug["name"]]
    if drug["atc_category"]:
        parts.append(f"ATC category: {drug['atc_category']}")
    if drug["smiles"]:
        parts.append(f"SMILES: {drug['smiles']}")
    if drug["iupac_name"]:
        parts.append(f"IUPAC name: {drug['iupac_name']}")
    if drug["description"]:
        parts.append(f"Description: {drug['description']}")
    return ". ".join(parts)


def load_existing(path: Path) -> tuple[dict[int, np.ndarray], str | None]:
    if not path.exists():
        return {}, None
    data = np.load(path, allow_pickle=True)
    nums = data["ddinter_nums"]
    vecs = data["embeddings"]
    tag = str(data["model_tag"][0]) if "model_tag" in data else None
    return {int(n): v for n, v in zip(nums, vecs)}, tag


def batched(items, n):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def embed_openai(todo: list[int], drugs: dict, model: str, batch_size: int) -> dict[int, np.ndarray]:
    try:
        from openai import OpenAI
    except ImportError:
        print("Missing dependency: pip install -r requirements-ml.txt", file=sys.stderr)
        raise

    if not os.environ.get("OPENAI_API_KEY"):
        print("[error] OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()
    results: dict[int, np.ndarray] = {}
    for batch_nums in batched(todo, batch_size):
        texts = [combined_text(drugs[n]) for n in batch_nums]
        resp = client.embeddings.create(model=model, input=texts)
        for num, item in zip(batch_nums, resp.data):
            results[num] = np.array(item.embedding, dtype=np.float32)
        print(f"[info] embedded {len(results):,}/{len(todo):,}")
        time.sleep(0.1)
    return results


def _mean_pool_embed(texts: list[str], tokenizer, encoder, batch_size: int):
    import torch

    out_vecs = []
    with torch.no_grad():
        for batch in batched(texts, batch_size):
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            out = encoder(**tokens)
            hidden = out.last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            out_vecs.extend(v.numpy().astype(np.float32) for v in pooled)
    return out_vecs


def embed_local(todo: list[int], drugs: dict, smiles_model: str, text_model: str, batch_size: int) -> dict[int, np.ndarray]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("Missing dependency: pip install -r requirements-ml.txt", file=sys.stderr)
        raise

    print(f"[info] Loading SMILES model '{smiles_model}' (first run downloads weights)...")
    smiles_tokenizer = AutoTokenizer.from_pretrained(smiles_model)
    smiles_encoder = AutoModel.from_pretrained(smiles_model)
    smiles_encoder.eval()
    smiles_dim = smiles_encoder.config.hidden_size

    print(f"[info] Loading text model '{text_model}' (first run downloads weights)...")
    text_tokenizer = AutoTokenizer.from_pretrained(text_model)
    text_encoder = AutoModel.from_pretrained(text_model)
    text_encoder.eval()

    # Drugs with no PubChem SMILES fall back to a zero vector for that half
    # of the feature — there's no structure to embed, but their name/ATC
    # text signal still comes through via the second half.
    with_smiles = [n for n in todo if drugs[n]["smiles"]]
    without_smiles = len(todo) - len(with_smiles)
    if without_smiles:
        print(f"[info] {without_smiles:,}/{len(todo):,} drugs have no SMILES — "
              "using zero-filled structure features for those (name/ATC-only signal).")

    print(f"[info] Embedding {len(with_smiles):,} SMILES strings...")
    smiles_texts = [drugs[n]["smiles"] for n in with_smiles]
    smiles_vecs = _mean_pool_embed(smiles_texts, smiles_tokenizer, smiles_encoder, batch_size) if smiles_texts else []
    smiles_vec_by_num = dict(zip(with_smiles, smiles_vecs))

    print(f"[info] Embedding {len(todo):,} name/ATC/IUPAC/description strings...")
    info_texts = [info_text(drugs[n]) for n in todo]
    text_vecs = _mean_pool_embed(info_texts, text_tokenizer, text_encoder, batch_size)

    results: dict[int, np.ndarray] = {}
    for num, t_vec in zip(todo, text_vecs):
        s_vec = smiles_vec_by_num.get(num, np.zeros(smiles_dim, dtype=np.float32))
        results[num] = np.concatenate([s_vec, t_vec])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["openai", "local"], default="local")
    parser.add_argument("--model", default=None,
                         help="OpenAI model, or the local SMILES model. Defaults: "
                              f"'{DEFAULT_MODEL}' (openai) / '{DEFAULT_SMILES_MODEL}' (local)")
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL,
                         help="Local-only: general text model for name/ATC/IUPAC/description")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    if args.backend == "openai":
        model = args.model or DEFAULT_MODEL
        model_tag = f"openai:{model}(combined)"
        batch_size = args.batch_size or 100
    else:
        model = args.model or DEFAULT_SMILES_MODEL
        model_tag = f"local:{model}+{args.text_model}(concat)"
        batch_size = args.batch_size or 32

    if not SMILES_CSV.exists():
        print(f"[error] {SMILES_CSV} not found — run fetch_smiles.py first.", file=sys.stderr)
        sys.exit(1)

    drugs = load_drug_data(SMILES_CSV)
    print(f"[info] {len(drugs):,} drugs with cached SMILES")

    existing, existing_tag = load_existing(EMBEDDINGS_NPZ)
    if existing and existing_tag and existing_tag != model_tag:
        print(f"[warn] Existing embeddings were built with '{existing_tag}', "
              f"not '{model_tag}'. Ignoring cache and re-embedding all drugs.", file=sys.stderr)
        existing = {}

    todo = [num for num in drugs if num not in existing]
    print(f"[info] {len(existing):,} already embedded, {len(todo):,} to embed (backend={args.backend})")

    if todo:
        new_results = (
            embed_openai(todo, drugs, model, batch_size)
            if args.backend == "openai"
            else embed_local(todo, drugs, model, args.text_model, batch_size)
        )
    else:
        new_results = {}

    results = dict(existing)
    results.update(new_results)

    if not results:
        print("[warn] No embeddings produced — nothing to save.", file=sys.stderr)
        return

    nums = np.array(sorted(results.keys()), dtype=np.int64)
    matrix = np.stack([results[n] for n in nums])
    np.savez(EMBEDDINGS_NPZ, ddinter_nums=nums, embeddings=matrix, model_tag=np.array([model_tag]))

    manifest = {
        "backend": args.backend,
        "model": model,
        "text_model": args.text_model if args.backend == "local" else None,
        "dim": int(matrix.shape[1]),
        "count": int(len(nums)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[info] Saved {len(nums):,} embeddings ({matrix.shape[1]}-dim) to {EMBEDDINGS_NPZ}")


if __name__ == "__main__":
    main()

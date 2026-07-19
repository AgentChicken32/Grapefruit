# DDI Prediction Pipeline

Offline batch pipeline that augments `backend/interactions.csv` with
AI-*predicted* drug-drug interactions, inspired by
[DDI-LLM](https://github.com/sshaghayeghs/DDI-LLM): embed each drug's
chemical structure and metadata with an LLM, use the embeddings as node
features in a GCN trained for link prediction on the known DDInter graph,
then score all currently-unconnected pairs.

Each drug is represented by four fields (`smiles_cache.csv`): **SMILES**
(structure), **ATC therapeutic category** (derived locally from
`interactions.csv` — best-effort majority vote, since DDInter's category
column describes an interaction row rather than a single drug), **IUPAC
name**, and **PubChem description**. SMILES and the natural-language
fields are embedded separately and concatenated (see step 2) rather than
mashed into one string, since a SMILES-specific tokenizer garbles English
text and vice versa.

**SMILES is optional.** Only drugs with a name-match to a PubChem CID
(currently ~907 of 2,265) get a structure; the rest still get a row with
name + ATC category, and fall back to a zero-filled structure half of the
feature vector plus real text signal for the other half — so the
prediction graph can cover all 2,265 drugs, not just the chemically-resolved
subset, at the cost of weaker signal for the drugs without SMILES.

**This is not part of the running FastAPI server.** Run it manually,
whenever you want to (re)generate predictions. Output never touches or
overwrites `interactions.csv` — it produces a separate
`backend/predicted_interactions.csv`, which `main.py` loads into its own
`predicted_interactions` table and the frontend renders with a distinct
"AI-predicted, not clinically verified" advisory badge.

## Setup

```bash
cd backend/ddi_predict
pip install -r requirements-ml.txt
export OPENAI_API_KEY=sk-...
```

## Steps (run in order)

```bash
# 1. Resolve DDInter drugs -> PubChem CID -> SMILES + IUPAC name + description,
#    plus a locally-derived ATC category. Cached and resumable per-field.
python fetch_smiles.py
#   --limit N       only process the first N drugs needing any field (smoke test)
#   --sleep 0.25     delay between PubChem requests

# 2. Embed each cached drug (cached, resumable). Two backends:
python embed_smiles.py --backend openai   # needs OPENAI_API_KEY + billing/quota
#   one combined string (name+ATC+SMILES+IUPAC+description) per drug
#   --model text-embedding-3-small  --batch-size 100
python embed_smiles.py --backend local    # free, offline, downloads weights once
#   SMILES -> ChemBERTa, name+ATC+IUPAC+description -> bert-base-uncased,
#   concatenated per drug
#   --model seyonec/ChemBERTa-zinc-base-v1  --text-model bert-base-uncased  --batch-size 32

# 3. Train the GCN link-prediction model on the known DDInter graph
python train_link_predictor.py
#   --epochs 100  --hidden 128  --out-dim 64  --lr 0.01

# 4. Score all unconnected pairs and write predicted_interactions.csv
python predict_new_interactions.py
#   --threshold 0.90     minimum predicted probability to keep a pair
#   --max-per-drug 20    cap kept candidates per drug (keeps the list reviewable)
```

Re-run from step 1 whenever `interactions.csv` changes materially. Steps 1
and 2 cache their results (`smiles_cache.csv`, `embeddings.npz`) and only
fetch/embed drugs that are missing, so re-runs are cheap.

## Design notes

- Only drugs with a resolvable PubChem SMILES are included in the
  prediction graph. IUPAC name and description are best-effort enrichment
  on top — not every PubChem CID has a description, and that's fine, the
  text field just ends up shorter for those drugs.
- Predictions are capped per-drug and thresholded to keep the output a
  reviewable, high-confidence candidate list rather than a firehose.
- `backend/main.py` never lets predicted data influence risk scores or
  coverage percentages — it's purely an additive, clearly-labeled
  annotation on top of verified DDInter data.

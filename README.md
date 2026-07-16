# Drug Interaction Risk Scorer (FISH-Drugs)

## Project Structure

```
FISH-Drugs/
├── backend/
│   ├── main.py                  # FastAPI app (reads from Supabase Postgres)
│   ├── schema.sql               # Postgres schema
│   ├── migrate_to_supabase.py   # One-time/re-runnable data import (see Cloud Deployment)
│   ├── build_drug_mapping.py    # Regenerates cid_to_ddinter.csv from Supabase + Drug_SE_DB
│   ├── interactions.csv         # Drug interaction data (local only, not committed)
│   └── requirements.txt
└── frontend/
    └── src/
        └── App.jsx              # Single-file React frontend (Vite project)
```

---

## CSV Format

Each row must have at least **5 columns**. A 7-column format is also supported (header row is auto-detected and skipped):

```
drug_a_id, drug_a_name, drug_b_id, drug_b_name, interaction_strength [, mechanism [, category]]
```

| Column | Field | Notes |
|--------|-------|-------|
| 1 | `drug_a_id` | Unique drug identifier |
| 2 | `drug_a_name` | Display name |
| 3 | `drug_b_id` | Unique drug identifier |
| 4 | `drug_b_name` | Display name |
| 5 | `interaction_strength` | Numeric value (any consistent scale) |
| 6 | `mechanism` | Short text describing the mechanism (e.g. `Absorption`, `Metabolism`, `Synergy`). Use `Unknown` when unknown — this value is treated as absent. |
| 7 | `category` | Ignored (reserved for future use) |

Example:
```
DDInter001,Warfarin,DDInter002,Aspirin,0.85,Synergy,Blood
DDInter001,Warfarin,DDInter003,Ibuprofen,0.72,Unknown,Blood
```

**Parsing rules:**
- Rows with non-numeric strength or fewer than 5 columns are silently skipped.
- Mechanism values that are `Unknown`, blank, or purely numeric (malformed rows) are stored as `NULL`.
- Column 7 (category) is parsed but not stored or used.
- Loaded into Supabase by `migrate_to_supabase.py` (see Cloud Deployment below), not rebuilt by the app itself. Re-run that script after changing the CSV.

### Drug-food interactions

A CSV with a header row and the following columns:

```
Severity level, Food name, Description, Management, Mechanism, References, drug_id
```

- `Severity level` must be an integer (1–5). Rows where it isn't a valid
  integer (e.g. `"No matching records"`) are skipped.
- `drug_id` should match the drug ID values used in the interactions CSV
  (e.g. `DDInter1075`).
- `References` is read but not currently surfaced by the API.
- Loaded into Supabase by `migrate_to_supabase.py`. If the file is missing at
  migration time, food-interaction lookups simply return empty results.

### Drug-disease interactions

A CSV with a header row and the following columns:

```
Severity level, Disease name, Text, drug_id
```

- `Severity level` must be an integer (1–5). Rows where it isn't a valid
  integer (e.g. `"No matching records"`) are skipped.
- `drug_id` should match the drug ID values used in the interactions CSV
  (e.g. `DDInter1075`).
- `Text` describes the risk/contraindication for that disease.
- Loaded into Supabase by `migrate_to_supabase.py`. If the file is missing at
  migration time, disease-interaction lookups simply return empty results.

---

## Backend Setup

Requires a Supabase project already populated via `migrate_to_supabase.py`
(see Cloud Deployment below) - the app only reads from Postgres, it doesn't
build any local database.

```bash
cd backend
pip install -r requirements.txt

cp .env.example .env
# edit .env and set DATABASE_URL to your Supabase connection string

# Optional: export SIM_CUTOFF=0.90  # similarity threshold (default 0.90)

uvicorn main:app --reload --port 8000
```

Server is ready at `http://localhost:8000`, with all routes under `/api`.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Returns DB interaction count |
| GET | `/api/search?q=warfarin&limit=8` | Autocomplete drug search (min 2 chars) |
| GET | `/api/drugs/{drug_id}/side_effects` | Side effects for a single drug |
| POST | `/api/regime/risk` | Full regime risk analysis |

#### POST `/api/regime/risk`

**Request:**
```json
{ "drug_ids": ["DDInter001", "DDInter002", "DDInter003"] }
```
Drug IDs **or** drug names are accepted (case-insensitive).

**Response (abbreviated):**
```json
{
  "drugs": [
    { "id": "DDInter001", "name": "Warfarin",  "avg_strength": 0.785, "risk": 0.785 },
    { "id": "DDInter002", "name": "Aspirin",   "avg_strength": 0.610, "risk": 0.610 }
  ],
  "total_risk": 1.395,
  "normalized_risk": 0.698,
  "populated_edges": 1,
  "possible_edges": 1,
  "coverage_pct": 100.0,
  "unknown_drugs": [],
  "pair_scores": [
    {
      "drug_a_id": "DDInter001", "drug_a_name": "Warfarin",
      "drug_b_id": "DDInter002", "drug_b_name": "Aspirin",
      "score": 0.72,
      "mechanism": "Synergy"
    }
  ],
  "similar_replacements": [
    {
      "drug_id": "DDInter001",
      "drug_name": "Warfarin",
      "original_interaction_count": 142,
      "replacements": [
        { "id": "DDInter099", "name": "Acenocoumarol", "score": 0.94, "interaction_count": 138 }
      ]
    }
  ]
}
```

---

## Risk Model

### Per-drug risk
Each drug's risk score is the **average interaction strength with the other drugs currently in the regime** (not all interactions in the database). Drugs with no recorded interactions with any regime partner contribute `0` and are shown with `avg_strength: null`.

### Regime risk
The **Regime Risk** displayed in the UI is the average of all individual drug risk scores:

```
regime_risk = sum(drug risks) / number_of_drugs
```

### Coverage
Coverage is the percentage of unique drug pairs in the regime that have at least one recorded interaction entry in the database.

### Matching scores (similarity)
Pairwise drug similarity uses the **Sørensen-Dice coefficient** over shared interaction neighbourhoods:

```
Score(A, B) = 2 × |neighbours(A) ∩ neighbours(B)| / (|neighbours(A)| + |neighbours(B)|)
```

Scores are precomputed for all drug pairs once by `migrate_to_supabase.py` and stored in `matching_scores` - not recomputed by the app itself. The "Similar Drug Replacements" section lists drugs outside the regime with a matching score ≥ the `SIM_CUTOFF` threshold (default 90%), sorted by total interaction count.

### Food interactions

For every drug in the regime, the API looks up any known drug-food
interactions and returns them sorted by severity (highest first). Each
entry includes the food name, a severity score (1–5), a description of the
interaction, the recommended management/mitigation, and the underlying
mechanism (e.g. absorption, metabolism). Drugs with no known food
interactions return an empty `foods` list.

### Disease interactions

For every drug in the regime, the API also looks up any known
drug-disease (contraindication) warnings and returns them sorted by
severity (highest first). Each entry includes the disease/condition name,
a severity score (1–5), and a text description of the risk. Drugs with no
known disease interactions return an empty `diseases` list.

---

## Frontend Setup

The frontend is a Vite + React project located in `frontend/`. Dependencies are listed in `frontend/package.json`.

```bash
cd frontend
npm install
npm run dev        # dev server at http://localhost:5173
npm run build      # production build → frontend/dist/
```

`API_BASE` (top of `src/App.jsx`) reads the `VITE_API_BASE` env var, falling back to `http://localhost:8000/api` for local dev. See `frontend/.env.example`.

### UI Sections

| Section | Description |
|---------|-------------|
| **Drug input bar** | Tag-style multi-drug selector with live autocomplete. Click outside or select a drug to close the dropdown. Backspace removes the last tag. |
| **Regime Risk** | Summary cards: overall risk score, DB coverage %, and drug count. |
| **Individual Drug Risk** | Per-drug risk bar (avg interaction strength with regime partners only). |
| **Pairwise Matching Scores** | Sørensen-Dice similarity for each pair within the regime, plus the interaction mechanism where known. |
| **Similar Drug Replacements** | For each regime drug, lists similar alternatives (score ≥ 90%) with their total interaction counts compared to the original drug's count. |

---

## Cloud Deployment (Vercel + Supabase)

The app is hosted on Vercel straight from GitHub, with Supabase (Postgres) as
the only datastore - `main.py` has no local-database fallback, `DATABASE_URL`
is required unconditionally, whether running locally or deployed.

### 1. Create a Supabase project

In the [Supabase dashboard](https://supabase.com/dashboard), create a new
project and wait for it to finish provisioning. Then go to
**Project Settings -> Database -> Connection string** and copy the
**Transaction pooler** string (port `6543`) - it's the one safe to use from
many short-lived connections (a one-off script today, serverless functions
later), unlike the direct connection on port `5432`.

### 2. Import the data into Supabase

```bash
cd backend
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste in your DATABASE_URL

python migrate_to_supabase.py
```

This applies [`backend/schema.sql`](backend/schema.sql) and loads, from
scratch each run (`TRUNCATE` then reload - safe to re-run after the source
files change):

- `interactions.csv` -> `interactions`
- (computed) Sorensen-Dice similarity -> `matching_scores`
- `drug_food.csv` -> `food_interactions`
- `drug_disease.csv` -> `disease_interactions`
- `cid_to_ddinter.csv` + `Drug_SE_DB/meddra_freq.tsv/meddra_freq.tsv` -> `side_effects`

None of these source files are committed to git (see `.gitignore`), so this
script only needs to be run from a machine that has them locally - it is not
part of the Vercel build.

### 3. Deploy to Vercel

[`vercel.json`](vercel.json) defines two Vercel **Services** in one project:
`frontend` (Vite, builds `frontend/`) and `backend` (FastAPI, `backend/main.py`'s
`app`), with a top-level rewrite sending `/api/(.*)` to the backend service and
everything else to the frontend. Both share one domain, so the API is
same-origin - no CORS involved in production.

Import the GitHub repo in Vercel as-is (repo root). In the Vercel project's
environment variables, set:
- `DATABASE_URL` - the same Supabase connection string used locally, for the
  `backend` service.
- `VITE_API_BASE=/api` - for the `frontend` service (see `frontend/.env.example`).

---

## Performance Notes

- Postgres with indexes on both drug ID columns handles 200k+ rows in milliseconds for typical queries.
- The matching-score precomputation is the slow step (O(n²) over all drugs) - it runs once in `migrate_to_supabase.py`, not on every app start.
- `/api/regime/risk`'s replacement-suggestion logic batches all of its lookups (interaction edges, SE burdens, food/disease/side-effect data, matching scores) into a small, fixed number of queries per regime, regardless of how many replacement candidates a drug has - important since each query is now a network round-trip to Supabase rather than a local disk read. See `bulk_*` functions in `main.py`.

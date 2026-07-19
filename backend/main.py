"""
Drug Interaction Risk Scorer — FastAPI Backend
Serves regime risk queries against Supabase Postgres (populated by
migrate_to_supabase.py — this app never writes to the DB, only reads).
"""

import itertools
import math
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from psycopg.rows import dict_row
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent / ".env")

DATABASE_URL = os.environ.get("DATABASE_URL")
SIMILARITY_CUTOFF = float(os.environ.get("SIM_CUTOFF", "0.90"))

MAX_DDI_STRENGTH = 3.0  # DDI interaction-strength scale used throughout

# Weight given to Method 1 (multiplicative compounding) vs Method 3
# (burden-weighted entropy) when blending the two into a drug's risk score.
# 1.0 = compounding only, 0.0 = entropy only.
RISK_METHOD_WEIGHT = float(os.environ.get("RISK_METHOD_WEIGHT", "0.5"))
COMPOUND_ALPHA = float(os.environ.get("COMPOUND_ALPHA", "0.5"))  # Method 1 SE-compounding exponent
ENTROPY_BETA   = float(os.environ.get("ENTROPY_BETA",   "0.3"))  # Method 3 entropy penalty

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _id_to_num(drug_id: str) -> int:
    """Strip the 'DDInter' prefix and return the integer drug number."""
    return int(drug_id[7:])  # 'DDInter' is 7 characters


def get_conn() -> psycopg.Connection:
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


# ---------------------------------------------------------------------------
# Bulk lookups
#
# Every function here takes a *set* of drugs (regime members, or every
# replacement candidate for a drug at once) and does one query for the whole
# set, instead of one query per drug/pair/candidate. This matters once every
# query is a network round-trip to Supabase rather than a local SQLite read.
# ---------------------------------------------------------------------------

def bulk_interaction_edges(
    conn: psycopg.Connection, drug_ids: list[str] | set[str],
) -> tuple[dict[str, dict[str, list[float]]], dict[tuple[str, str], str | None]]:
    """
    One query: every interaction row where both drugs are in drug_ids.

    A pair of drugs can have more than one row (different mechanism entries
    for the same two drugs are common in this dataset - ~19% of pairs have
    duplicates), so each neighbour maps to a *list* of strengths rather than
    a single value, preserving the same duplicate-counting the old per-pair
    queries had (each row counted once toward that neighbour's contribution
    to the risk formula).

    Returns (edges, mechanisms):
      - edges[a][b] is symmetric: edges[a][b] and edges[b][a] both exist.
        A pair only appears here if it has at least one verified interaction
        row - used elsewhere to distinguish "verified" from "AI-predicted".
      - mechanisms[(a, b)] picks one arbitrary mechanism per pair (matches
        the old single-row `LIMIT 1` lookup's "any one" semantics).
    """
    ids = list(drug_ids)
    num_to_id = {_id_to_num(did): did for did in ids}
    nums = list(num_to_id.keys())
    edges: dict[str, dict[str, list[float]]] = {did: {} for did in ids}
    mechanisms: dict[tuple[str, str], str | None] = {}
    if len(nums) < 2:
        return edges, mechanisms

    placeholders = ",".join(["%s"] * len(nums))
    rows = conn.execute(
        f"""
        SELECT drug_a_num, drug_b_num, strength, mechanism
        FROM interactions
        WHERE drug_a_num IN ({placeholders}) AND drug_b_num IN ({placeholders})
        """,
        nums + nums,
    ).fetchall()
    for row in rows:
        a_id = num_to_id.get(row["drug_a_num"])
        b_id = num_to_id.get(row["drug_b_num"])
        if a_id is None or b_id is None or a_id == b_id:
            continue
        edges[a_id].setdefault(b_id, []).append(row["strength"])
        edges[b_id].setdefault(a_id, []).append(row["strength"])
        mechanisms.setdefault((a_id, b_id), row["mechanism"])
        mechanisms.setdefault((b_id, a_id), row["mechanism"])
    return edges, mechanisms


def _neighbor_strengths(
    edges_for_drug: dict[str, list[float]], member_set: set[str] | None = None,
) -> list[float]:
    """Flatten a drug's {neighbour: [strengths]} edges into one list, optionally restricted to member_set."""
    items = edges_for_drug.items() if member_set is None else (
        (other, lst) for other, lst in edges_for_drug.items() if other in member_set
    )
    return [s for _, lst in items for s in lst]


def regime_se_burdens(conn: psycopg.Connection, drug_ids: list[str] | set[str]) -> dict[str, list[float]]:
    """
    One query for the whole set: each drug's severity-weighted side-effect
    burdens. Each value is freq_upper * (severity / 5) when that side effect
    has a severity score (see score_se_severity.py), else plain freq_upper -
    same per-item shape compounding_risk/entropy_risk already expect, just
    weighted before it gets there.
    """
    ids = list(drug_ids)
    result: dict[str, list[float]] = {did: [] for did in ids}
    num_to_id = {_id_to_num(did): did for did in ids}
    nums = list(num_to_id.keys())
    if not nums:
        return result
    placeholders = ",".join(["%s"] * len(nums))
    rows = conn.execute(
        f"""
        SELECT ddinter_num, freq_upper, severity
        FROM side_effects
        WHERE ddinter_num IN ({placeholders}) AND freq_upper IS NOT NULL
        """,
        nums,
    ).fetchall()
    for row in rows:
        did = num_to_id.get(row["ddinter_num"])
        if did is None:
            continue
        burden = row["freq_upper"] * (row["severity"] / 5.0) if row["severity"] is not None else row["freq_upper"]
        result[did].append(burden)
    return result


def bulk_matching_scores(conn: psycopg.Connection, drug_ids: list[str] | set[str]) -> dict[tuple[str, str], float]:
    """One query: Sorensen-Dice matching score for every pair within drug_ids (symmetric)."""
    ids = list(drug_ids)
    num_to_id = {_id_to_num(did): did for did in ids}
    nums = list(num_to_id.keys())
    scores: dict[tuple[str, str], float] = {}
    if len(nums) < 2:
        return scores
    placeholders = ",".join(["%s"] * len(nums))
    rows = conn.execute(
        f"""
        SELECT drug_a_num, drug_b_num, score
        FROM matching_scores
        WHERE drug_a_num IN ({placeholders}) AND drug_b_num IN ({placeholders})
        """,
        nums + nums,
    ).fetchall()
    for row in rows:
        a_id = num_to_id.get(row["drug_a_num"])
        b_id = num_to_id.get(row["drug_b_num"])
        if a_id is None or b_id is None:
            continue
        scores[(a_id, b_id)] = row["score"]
        scores[(b_id, a_id)] = row["score"]
    return scores


def bulk_candidate_info(conn: psycopg.Connection, drug_nums: list[int]) -> dict[int, tuple[int, str]]:
    """One query: total interaction count + a display name, per drug number."""
    if not drug_nums:
        return {}
    placeholders = ",".join(["%s"] * len(drug_nums))
    rows = conn.execute(
        f"""
        SELECT num, COUNT(*) AS cnt, MAX(name) AS name FROM (
            SELECT drug_a_num AS num, drug_a_name AS name FROM interactions WHERE drug_a_num IN ({placeholders})
            UNION ALL
            SELECT drug_b_num AS num, drug_b_name AS name FROM interactions WHERE drug_b_num IN ({placeholders})
        ) t
        GROUP BY num
        """,
        drug_nums + drug_nums,
    ).fetchall()
    return {row["num"]: (row["cnt"], row["name"]) for row in rows}


def bulk_food_interactions(conn: psycopg.Connection, drug_nums: list[int]) -> dict[int, list[dict]]:
    result: dict[int, list[dict]] = {n: [] for n in drug_nums}
    if not drug_nums:
        return result
    placeholders = ",".join(["%s"] * len(drug_nums))
    rows = conn.execute(
        f"""
        SELECT drug_num, food_name, severity, description, management, mechanism
        FROM food_interactions
        WHERE drug_num IN ({placeholders})
        ORDER BY drug_num, severity DESC, food_name ASC
        """,
        drug_nums,
    ).fetchall()
    for row in rows:
        result[row["drug_num"]].append({
            "food_name":   row["food_name"],
            "severity":    row["severity"],
            "description": row["description"],
            "management":  row["management"],
            "mechanism":   row["mechanism"],
        })
    return result


def bulk_disease_interactions(conn: psycopg.Connection, drug_nums: list[int]) -> dict[int, list[dict]]:
    result: dict[int, list[dict]] = {n: [] for n in drug_nums}
    if not drug_nums:
        return result
    placeholders = ",".join(["%s"] * len(drug_nums))
    rows = conn.execute(
        f"""
        SELECT drug_num, disease_name, severity, text
        FROM disease_interactions
        WHERE drug_num IN ({placeholders})
        ORDER BY drug_num, severity DESC, disease_name ASC
        """,
        drug_nums,
    ).fetchall()
    for row in rows:
        result[row["drug_num"]].append({
            "disease_name": row["disease_name"],
            "severity":     row["severity"],
            "text":         row["text"],
        })
    return result


def bulk_side_effects(conn: psycopg.Connection, drug_nums: list[int], limit: int = 50) -> dict[int, list[dict]]:
    """Side effects per drug, ordered by descending frequency, capped at `limit` per drug."""
    result: dict[int, list[dict]] = {n: [] for n in drug_nums}
    if not drug_nums:
        return result
    placeholders = ",".join(["%s"] * len(drug_nums))
    rows = conn.execute(
        f"""
        SELECT ddinter_num, se_name, freq_lower, freq_upper, freq_label, severity
        FROM side_effects
        WHERE ddinter_num IN ({placeholders})
          AND (freq_upper IS NULL OR freq_upper > 0)
          AND (freq_lower IS NULL OR freq_lower > 0)
        ORDER BY
          ddinter_num,
          CASE freq_label
            WHEN 'very common'   THEN 10
            WHEN 'common'        THEN 20
            WHEN 'frequent'      THEN 30
            WHEN 'uncommon'      THEN 40
            WHEN 'infrequent'    THEN 50
            WHEN 'rare'          THEN 60
            WHEN 'very rare'     THEN 70
            ELSE 80
          END,
          freq_lower DESC NULLS LAST,
          se_name ASC
        """,
        drug_nums,
    ).fetchall()
    for row in rows:
        bucket = result[row["ddinter_num"]]
        if len(bucket) < limit:
            bucket.append({
                "se_name":    row["se_name"],
                "freq_lower": row["freq_lower"],
                "freq_upper": row["freq_upper"],
                "freq_label": row["freq_label"],
                "severity":   row["severity"],
            })
    return result


# ---------------------------------------------------------------------------
# Predicted (AI-generated) interactions
#
# These come from the offline pipeline in backend/ddi_predict/ — a GCN link
# predictor trained on the verified DDInter graph, using SMILES-based drug
# embeddings as node features (see ddi_predict/README.md). They are NOT
# clinically verified and must never influence strength/risk/coverage
# calculations, which stay derived solely from the `interactions` table.
# Loaded into Postgres by migrate_to_supabase.py, same as everything else.
# ---------------------------------------------------------------------------

def _predicted_advisory(method: str) -> str:
    return (
        f"Predicted by an AI model ({method}, trained on the known DDInter "
        "interaction graph using SMILES-based drug embeddings) — not a "
        "clinically verified interaction."
    )


def bulk_predicted_interactions(
    conn: psycopg.Connection, drug_nums: list[int],
) -> dict[tuple[int, int], dict]:
    """One query: best (highest-confidence) predicted interaction per unordered pair within drug_nums."""
    result: dict[tuple[int, int], dict] = {}
    if len(drug_nums) < 2:
        return result
    placeholders = ",".join(["%s"] * len(drug_nums))
    rows = conn.execute(
        f"""
        SELECT drug_a_num, drug_b_num, confidence, method
        FROM predicted_interactions
        WHERE drug_a_num IN ({placeholders}) AND drug_b_num IN ({placeholders})
        ORDER BY confidence DESC
        """,
        drug_nums + drug_nums,
    ).fetchall()
    for row in rows:
        key = tuple(sorted((row["drug_a_num"], row["drug_b_num"])))
        if key not in result:  # first row per pair wins - ORDER BY confidence DESC
            result[key] = {"confidence": row["confidence"], "method": row["method"]}
    return result


# ---------------------------------------------------------------------------
# Similarity-based replacement suggestions
# ---------------------------------------------------------------------------

def find_similar_replacements(
    conn: psycopg.Connection,
    drug_id: str,
    regime_ids: set[str],
    cutoff: float = SIMILARITY_CUTOFF,
) -> list[dict]:
    drug_num = _id_to_num(drug_id)
    rows = conn.execute(
        """
        SELECT 'DDInter' || ms.drug_b_num AS cand_id, ms.score
        FROM matching_scores ms
        WHERE ms.drug_a_num = %s AND ms.score >= %s
        UNION ALL
        SELECT 'DDInter' || ms.drug_a_num AS cand_id, ms.score
        FROM matching_scores ms
        WHERE ms.drug_b_num = %s AND ms.score >= %s
        """,
        (drug_num, cutoff, drug_num, cutoff),
    ).fetchall()

    candidates = [(row["cand_id"], row["score"]) for row in rows if row["cand_id"] not in regime_ids]
    if not candidates:
        return []

    cand_nums = [_id_to_num(cid) for cid, _ in candidates]
    info            = bulk_candidate_info(conn, cand_nums)
    foods_by_num    = bulk_food_interactions(conn, cand_nums)
    diseases_by_num = bulk_disease_interactions(conn, cand_nums)
    se_by_num       = bulk_side_effects(conn, cand_nums)

    results = []
    for cand_id, score in candidates:
        cand_num = _id_to_num(cand_id)
        cnt, name = info.get(cand_num, (0, cand_id))
        results.append({
            "id":                cand_id,
            "name":              name,
            "score":             score,
            "interaction_count": cnt,
            "foods":             foods_by_num.get(cand_num, []),
            "diseases":          diseases_by_num.get(cand_num, []),
            "side_effects":      se_by_num.get(cand_num, []),
        })

    results.sort(key=lambda r: r["interaction_count"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Risk calculation
# ---------------------------------------------------------------------------

def compounding_risk(
    ddi_strengths: list[float],
    se_burdens: list[float],
    alpha: float = COMPOUND_ALPHA,
) -> float:
    """
    Method 1 - Multiplicative Risk Compounding.
    Risk = MAX_DDI_STRENGTH * [1 - prod(1 - DDIi/MAX)] * [1 - prod(1 - SEj)]^alpha
    A channel with no data is dropped rather than treated as a verified
    zero, so a drug with SE data but no DDI partners (or vice versa) isn't
    forced to a total risk of 0.
    """
    ddi_bracket = None
    if ddi_strengths:
        survival = 1.0
        for s in ddi_strengths:
            survival *= (1.0 - s / MAX_DDI_STRENGTH)
        ddi_bracket = 1.0 - survival

    se_bracket = None
    if se_burdens:
        survival = 1.0
        for b in se_burdens:
            survival *= (1.0 - b)
        se_bracket = 1.0 - survival

    if ddi_bracket is None and se_bracket is None:
        return 0.0
    if ddi_bracket is None:
        return MAX_DDI_STRENGTH * (se_bracket ** alpha)
    if se_bracket is None:
        return MAX_DDI_STRENGTH * ddi_bracket
    return MAX_DDI_STRENGTH * ddi_bracket * (se_bracket ** alpha)


def entropy_risk(
    ddi_strengths: list[float],
    se_burdens: list[float],
    beta: float = ENTROPY_BETA,
) -> float:
    """
    Method 3 - Burden-Weighted Entropy Score.
    Each DDI strength and each SE burden (rescaled 0-1 -> 0-3) is treated as
    one event with magnitude b_k, weighted by its own share of total
    magnitude p_k = b_k / sum(b). Risk = sum(p_k * b_k) - beta * H(p), which
    reduces to sum(b_k^2) / sum(b_k) - beta * H(p).
    """
    magnitudes = [s for s in ddi_strengths if s > 0]
    magnitudes += [MAX_DDI_STRENGTH * b for b in se_burdens if b > 0]
    if not magnitudes:
        return 0.0
    total = sum(magnitudes)
    weighted_mean = sum(m * m for m in magnitudes) / total
    entropy = -sum((m / total) * math.log(m / total) for m in magnitudes)
    return max(0.0, weighted_mean - beta * entropy)


def blended_method_risk(
    ddi_strengths: list[float],
    se_burdens: list[float],
    weight: float = RISK_METHOD_WEIGHT,
) -> float:
    """Weighted average of Method 1 and Method 3, weight adjustable via RISK_METHOD_WEIGHT."""
    m1 = compounding_risk(ddi_strengths, se_burdens)
    m3 = entropy_risk(ddi_strengths, se_burdens)
    return weight * m1 + (1.0 - weight) * m3


def regime_risk_from_edges(
    member_ids: list[str],
    edges: dict[str, dict[str, list[float]]],
    burdens: dict[str, list[float]],
    weight: float = RISK_METHOD_WEIGHT,
) -> float:
    """
    Average blended risk over member_ids, using already-fetched edges/burdens
    (restricted to member_ids) instead of issuing fresh queries. Used to score
    hypothetical substitute regimes (one drug swapped for a replacement
    candidate) without a DB round-trip per candidate.
    """
    if not member_ids:
        return 0.0
    member_set = set(member_ids)
    risks = [
        blended_method_risk(
            _neighbor_strengths(edges.get(did, {}), member_set),
            burdens.get(did, []),
            weight=weight,
        )
        for did in member_ids
    ]
    return sum(risks) / len(risks)


def resolve_drug(conn: psycopg.Connection, query: str) -> dict | None:
    num: int | None = None
    if query.upper().startswith("DDINTER"):
        try:
            num = _id_to_num(query)
        except (ValueError, IndexError):
            pass

    if num is not None:
        row = conn.execute(
            """
            SELECT 'DDInter' || drug_a_num AS id, drug_a_name AS name
            FROM interactions WHERE drug_a_num = %s LIMIT 1
            """,
            (num,),
        ).fetchone()
        if not row:
            row = conn.execute(
                """
                SELECT 'DDInter' || drug_b_num AS id, drug_b_name AS name
                FROM interactions WHERE drug_b_num = %s LIMIT 1
                """,
                (num,),
            ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT 'DDInter' || drug_a_num AS id, drug_a_name AS name
            FROM interactions WHERE LOWER(drug_a_name) = LOWER(%s) LIMIT 1
            """,
            (query,),
        ).fetchone()
        if not row:
            row = conn.execute(
                """
                SELECT 'DDInter' || drug_b_num AS id, drug_b_name AS name
                FROM interactions WHERE LOWER(drug_b_name) = LOWER(%s) LIMIT 1
                """,
                (query,),
            ).fetchone()
    return dict(row) if row else None


def search_drugs(conn: psycopg.Connection, query: str, limit: int = 10) -> list[dict]:
    pattern = f"%{query}%"
    # If query is purely numeric, also match by DDInter number
    num_filter = query.strip().lstrip("0") or "0"
    is_numeric = query.strip().isdigit()
    rows = conn.execute(
        """
        SELECT DISTINCT 'DDInter' || drug_a_num AS id, drug_a_name AS name
        FROM interactions
        WHERE LOWER(drug_a_name) LIKE LOWER(%s)
           OR (%s AND CAST(drug_a_num AS TEXT) = %s)
        UNION
        SELECT DISTINCT 'DDInter' || drug_b_num AS id, drug_b_name AS name
        FROM interactions
        WHERE LOWER(drug_b_name) LIKE LOWER(%s)
           OR (%s AND CAST(drug_b_num AS TEXT) = %s)
        LIMIT %s
        """,
        (pattern, is_numeric, num_filter, pattern, is_numeric, num_filter, limit),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Drug Interaction Risk Scorer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RegimeRequest(BaseModel):
    drug_ids:           list[str]
    risk_method_weight: float | None = None  # override RISK_METHOD_WEIGHT (0–1)


class DrugRisk(BaseModel):
    id:           str
    name:         str
    avg_strength: float | None
    se_burden:    float | None  # avg side-effect frequency, 0-1 scale
    risk:         float         # blended score on 0-3 scale


class PairScore(BaseModel):
    drug_a_id:   str
    drug_a_name: str
    drug_b_id:   str
    drug_b_name: str
    score:       float | None
    mechanism:   str | None
    is_predicted:         bool         = False
    predicted_confidence: float | None = None
    advisory:             str | None   = None


class FoodInteraction(BaseModel):
    food_name:   str
    severity:    int
    description: str
    management:  str
    mechanism:   str


class DrugFoodInteractions(BaseModel):
    drug_id:   str
    drug_name: str
    foods:     list[FoodInteraction]


class DiseaseInteraction(BaseModel):
    disease_name: str
    severity:     int
    text:         str


class DrugDiseaseInteractions(BaseModel):
    drug_id:   str
    drug_name: str
    diseases:  list[DiseaseInteraction]


class SideEffect(BaseModel):
    se_name:    str
    freq_lower: float | None
    freq_upper: float | None
    freq_label: str | None
    severity:   int | None


class DrugSideEffects(BaseModel):
    drug_id:      str
    drug_name:    str
    side_effects: list[SideEffect]


class ReplacementCandidate(BaseModel):
    id:                str
    name:              str
    score:             float
    interaction_count: int
    foods:             list[FoodInteraction]
    diseases:          list[DiseaseInteraction]
    side_effects:      list[SideEffect]
    substitute_risk:   float | None


class DrugReplacements(BaseModel):
    drug_id:                    str
    drug_name:                  str
    original_interaction_count: int
    replacements:               list[ReplacementCandidate]


class RegimeResponse(BaseModel):
    drugs:                list[DrugRisk]
    total_risk:           float
    normalized_risk:      float
    risk_method_weight:   float
    similarity_cutoff:    float
    populated_edges:      int
    possible_edges:       int
    coverage_pct:         float
    unknown_drugs:        list[str]
    pair_scores:          list[PairScore]
    similar_replacements: list[DrugReplacements]
    food_interactions:    list[DrugFoodInteractions]
    disease_interactions: list[DrugDiseaseInteractions]
    side_effects:         list[DrugSideEffects]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    conn = get_conn()
    try:
        count = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()["count"]
    finally:
        conn.close()
    return {"status": "ok", "interactions": count}


@app.get("/api/search")
def search(q: str, limit: int = 10):
    if len(q) < 2:
        return []
    conn = get_conn()
    try:
        return search_drugs(conn, q, limit)
    finally:
        conn.close()


@app.get("/api/drugs/{drug_id}/side_effects", response_model=DrugSideEffects)
def drug_side_effects(drug_id: str, limit: int = 50):
    conn = get_conn()
    try:
        drug = resolve_drug(conn, drug_id)
        if not drug:
            raise HTTPException(status_code=404, detail=f"Drug '{drug_id}' not found")
        effects = bulk_side_effects(conn, [_id_to_num(drug["id"])], limit=limit)[_id_to_num(drug["id"])]
        return DrugSideEffects(
            drug_id=drug["id"],
            drug_name=drug["name"],
            side_effects=[SideEffect(**e) for e in effects],
        )
    finally:
        conn.close()


@app.post("/api/regime/risk", response_model=RegimeResponse)
def regime_risk(req: RegimeRequest):
    if not req.drug_ids:
        raise HTTPException(status_code=400, detail="drug_ids must not be empty")

    conn = get_conn()
    try:
        resolved: list[dict] = []
        unknown: list[str] = []

        for q in req.drug_ids:
            drug = resolve_drug(conn, q)
            if drug:
                if not any(d["id"] == drug["id"] for d in resolved):
                    resolved.append(drug)
            else:
                unknown.append(q)

        resolved_ids = [d["id"] for d in resolved]
        regime_nums = [_id_to_num(d["id"]) for d in resolved]
        eff_weight = req.risk_method_weight if req.risk_method_weight is not None else RISK_METHOD_WEIGHT

        # Batched: one query for every interaction edge (+ mechanism) among
        # regime drugs, one for SE burdens, one for matching scores, one for
        # AI-predicted interactions, one for original interaction counts --
        # covers drug_risks, pair_scores, and coverage without a query per pair.
        edges, mechanisms = bulk_interaction_edges(conn, resolved_ids)
        burdens           = regime_se_burdens(conn, resolved_ids)
        match_scores      = bulk_matching_scores(conn, resolved_ids)
        predicted         = bulk_predicted_interactions(conn, regime_nums)
        orig_counts       = bulk_candidate_info(conn, regime_nums)

        drug_risks: list[DrugRisk] = []
        for drug in resolved:
            strengths = _neighbor_strengths(edges[drug["id"]])
            drug_burdens = burdens[drug["id"]]
            avg    = sum(strengths) / len(strengths) if strengths else None
            burden = sum(drug_burdens) / len(drug_burdens) if drug_burdens else None
            risk_val = blended_method_risk(strengths, drug_burdens, weight=eff_weight)
            drug_risks.append(DrugRisk(
                id=drug["id"],
                name=drug["name"],
                avg_strength=avg,
                se_burden=round(burden, 4) if burden is not None else None,
                risk=round(risk_val, 4),
            ))

        total_risk = sum(d.risk for d in drug_risks)
        n = len(drug_risks)
        normalized_risk = total_risk / n if n > 0 else 0.0

        ids        = [d.id   for d in drug_risks]
        names      = {d.id: d.name for d in drug_risks}
        regime_set = set(ids)
        pairs      = list(itertools.combinations(ids, 2))
        possible   = len(pairs)
        populated  = sum(1 for a, b in pairs if b in edges.get(a, {}))
        coverage   = (populated / possible * 100) if possible > 0 else 0.0

        pair_scores: list[PairScore] = []
        for id_a, id_b in pairs:
            has_interaction = id_b in edges.get(id_a, {})
            mechanism = mechanisms.get((id_a, id_b))

            # AI-predicted interactions are purely additive: only surfaced
            # when there's no verified interaction row for this pair, and
            # never allowed to affect strength/risk/coverage calculations.
            is_predicted = False
            predicted_confidence = None
            advisory = None
            if not has_interaction:
                na, nb = _id_to_num(id_a), _id_to_num(id_b)
                pred = predicted.get(tuple(sorted((na, nb))))
                if pred:
                    is_predicted = True
                    predicted_confidence = round(pred["confidence"], 4)
                    advisory = _predicted_advisory(pred["method"])

            pair_scores.append(PairScore(
                drug_a_id=id_a,
                drug_a_name=names[id_a],
                drug_b_id=id_b,
                drug_b_name=names[id_b],
                score=match_scores.get((id_a, id_b)),
                mechanism=mechanism,
                is_predicted=is_predicted,
                predicted_confidence=predicted_confidence,
                advisory=advisory,
            ))

        similar_replacements: list[DrugReplacements] = []
        for drug in resolved:
            candidates = find_similar_replacements(conn, drug["id"], regime_set)
            orig_count = orig_counts.get(_id_to_num(drug["id"]), (0, drug["name"]))[0]

            # For each candidate, compute regime risk with this drug swapped
            # out -- batched: one query for the union of {base regime, every
            # candidate} instead of one query pair per candidate.
            base_ids      = [rid for rid in resolved_ids if rid != drug["id"]]
            candidate_ids = [c["id"] for c in candidates]
            union_ids     = list(set(base_ids) | set(candidate_ids))
            sub_edges, _  = bulk_interaction_edges(conn, union_ids)
            sub_burdens   = regime_se_burdens(conn, union_ids)

            replacement_objects: list[ReplacementCandidate] = []
            for c in candidates:
                sub_ids = base_ids + [c["id"]]
                sub_risk = regime_risk_from_edges(sub_ids, sub_edges, sub_burdens, weight=eff_weight)
                replacement_objects.append(ReplacementCandidate(
                    **c,
                    substitute_risk=round(sub_risk, 6) if sub_risk is not None else None,
                ))

            similar_replacements.append(DrugReplacements(
                drug_id=drug["id"],
                drug_name=drug["name"],
                original_interaction_count=orig_count,
                replacements=replacement_objects,
            ))

        # Main regime's own food/disease/side-effect displays, batched.
        foods_by_num    = bulk_food_interactions(conn, regime_nums)
        diseases_by_num = bulk_disease_interactions(conn, regime_nums)
        se_by_num       = bulk_side_effects(conn, regime_nums)

        food_interactions = [
            DrugFoodInteractions(
                drug_id=d["id"], drug_name=d["name"],
                foods=[FoodInteraction(**f) for f in foods_by_num.get(_id_to_num(d["id"]), [])],
            )
            for d in resolved
        ]
        disease_interactions = [
            DrugDiseaseInteractions(
                drug_id=d["id"], drug_name=d["name"],
                diseases=[DiseaseInteraction(**x) for x in diseases_by_num.get(_id_to_num(d["id"]), [])],
            )
            for d in resolved
        ]
        side_effects = [
            DrugSideEffects(
                drug_id=d["id"], drug_name=d["name"],
                side_effects=[SideEffect(**x) for x in se_by_num.get(_id_to_num(d["id"]), [])],
            )
            for d in resolved
        ]

        return RegimeResponse(
            drugs=drug_risks,
            total_risk=total_risk,
            normalized_risk=round(normalized_risk, 6),
            risk_method_weight=eff_weight,
            similarity_cutoff=SIMILARITY_CUTOFF,
            populated_edges=populated,
            possible_edges=possible,
            coverage_pct=round(coverage, 1),
            unknown_drugs=unknown,
            pair_scores=pair_scores,
            similar_replacements=similar_replacements,
            food_interactions=food_interactions,
            disease_interactions=disease_interactions,
            side_effects=side_effects,
        )
    finally:
        conn.close()

"""
Drug Interaction Risk Scorer — FastAPI Backend
Rebuilds SQLite DB from CSV on startup, then serves regime risk queries.
Computes and persists pairwise drug matching scores on first run.
"""

import csv
import itertools
import os
import sqlite3
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CSV_PATH          = os.environ.get("DRUG_CSV",     "interactions.csv")
FOOD_CSV_PATH     = os.environ.get("FOOD_CSV",     "drug_food.csv")
DISEASE_CSV_PATH  = os.environ.get("DISEASE_CSV",  "drug_disease.csv")
DB_PATH           = os.environ.get("DRUG_DB",      "interactions.db")
SIMILARITY_CUTOFF = float(os.environ.get("SIM_CUTOFF", "0.90"))
# Weight given to side-effect burden vs drug-drug interaction strength (0–1).
# 0.0 = interactions only, 1.0 = side effects only.
SE_WEIGHT = float(os.environ.get("SE_WEIGHT", "0.25"))

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def build_database(csv_path: str, db_path: str) -> None:
    """Drop and rebuild the interactions DB from a CSV file."""
    if not Path(csv_path).exists():
        print(f"[warn] CSV not found at '{csv_path}' — starting with empty DB.", file=sys.stderr)
        _init_schema(db_path)
        return

    conn = sqlite3.connect(db_path)
    try:
        _init_schema(db_path, conn=conn)
        cur = conn.cursor()

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            first = next(reader, None)
            if first and not _is_data_row(first):
                rows = reader
            else:
                rows = itertools.chain([first], reader) if first else reader

            batch = []
            for row in rows:
                if len(row) < 5:
                    continue
                drug_a_id, drug_a_name, drug_b_id, drug_b_name, strength = row[:5]
                try:
                    strength_f = float(strength)
                except ValueError:
                    continue
                # Column 6 (index 5): mechanism of interaction.
                # Treat missing, "Unknown", or purely numeric values as None.
                raw_mech = row[5].strip() if len(row) > 5 else ""
                mechanism: str | None = None
                if raw_mech and raw_mech.lower() != "unknown":
                    try:
                        float(raw_mech)   # numeric-only → malformed row
                    except ValueError:
                        mechanism = raw_mech
                batch.append((
                    _id_to_num(drug_a_id.strip()), drug_a_name.strip(),
                    _id_to_num(drug_b_id.strip()), drug_b_name.strip(),
                    strength_f, mechanism,
                ))
                if len(batch) >= 10_000:
                    cur.executemany(_INSERT_SQL, batch)
                    batch.clear()
            if batch:
                cur.executemany(_INSERT_SQL, batch)

        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        print(f"[info] Loaded {count:,} interactions into '{db_path}'.")
    finally:
        conn.close()


def _is_data_row(row: list[str]) -> bool:
    try:
        float(row[4])
        return True
    except (ValueError, IndexError):
        return False


_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    drug_a_num  INTEGER NOT NULL,
    drug_a_name TEXT    NOT NULL,
    drug_b_num  INTEGER NOT NULL,
    drug_b_name TEXT    NOT NULL,
    strength    REAL    NOT NULL,
    mechanism   TEXT
);
CREATE INDEX IF NOT EXISTS idx_a ON interactions(drug_a_num);
CREATE INDEX IF NOT EXISTS idx_b ON interactions(drug_b_num);
"""

_INSERT_SQL = """
INSERT INTO interactions (drug_a_num, drug_a_name, drug_b_num, drug_b_name, strength, mechanism)
VALUES (?, ?, ?, ?, ?, ?)
"""

_MATCHING_SCHEMA = """
CREATE TABLE IF NOT EXISTS matching_scores (
    drug_a_num INTEGER NOT NULL,
    drug_b_num INTEGER NOT NULL,
    score      REAL    NOT NULL,
    PRIMARY KEY (drug_a_num, drug_b_num)
);
"""


def _id_to_num(drug_id: str) -> int:
    """Strip the 'DDInter' prefix and return the integer drug number."""
    return int(drug_id[7:])  # 'DDInter' is 7 characters


def _init_schema(db_path: str, conn: sqlite3.Connection | None = None) -> None:
    close = conn is None
    if close:
        conn = sqlite3.connect(db_path)
    conn.executescript("DROP TABLE IF EXISTS interactions;" + _SCHEMA)
    conn.commit()
    if close:
        conn.close()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Matching score computation
# ---------------------------------------------------------------------------

def build_matching_scores(db_path: str) -> None:
    """
    Compute Sorensen-Dice matching scores for every pair of drugs and persist
    them into matching_scores.  Skipped if the table already has rows.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_MATCHING_SCHEMA)
        conn.commit()

        existing = conn.execute("SELECT COUNT(*) FROM matching_scores").fetchone()[0]
        if existing > 0:
            print(f"[info] Matching scores already present ({existing:,} rows) - skipping rebuild.")
            return

        print("[info] Building matching score lookup table...")

        rows = conn.execute(
            """
            SELECT drug_a_num AS id, drug_b_num AS neighbor FROM interactions
            UNION ALL
            SELECT drug_b_num AS id, drug_a_num AS neighbor FROM interactions
            """
        ).fetchall()

        neighbors: dict[int, set[int]] = {}
        for row in rows:
            drug_num     = row[0]
            neighbor_num = row[1]
            if drug_num not in neighbors:
                neighbors[drug_num] = set()
            neighbors[drug_num].add(neighbor_num)

        drug_nums   = list(neighbors.keys())
        total_pairs = len(drug_nums) * (len(drug_nums) - 1) // 2
        print(f"[info] {len(drug_nums):,} drugs -> {total_pairs:,} pairs to score.")

        batch = []
        BATCH_SIZE = 50_000

        for num_a, num_b in itertools.combinations(drug_nums, 2):
            na = neighbors[num_a]
            nb = neighbors[num_b]
            denom = len(na) + len(nb)
            score = (2 * len(na & nb) / denom) if denom > 0 else 0.0
            lo, hi = (num_a, num_b) if num_a < num_b else (num_b, num_a)
            batch.append((lo, hi, score))
            if len(batch) >= BATCH_SIZE:
                conn.executemany(
                    "INSERT OR IGNORE INTO matching_scores VALUES (?, ?, ?)", batch
                )
                conn.commit()
                batch.clear()

        if batch:
            conn.executemany(
                "INSERT OR IGNORE INTO matching_scores VALUES (?, ?, ?)", batch
            )
            conn.commit()

        final = conn.execute("SELECT COUNT(*) FROM matching_scores").fetchone()[0]
        print(f"[info] Matching score table built: {final:,} rows.")
    finally:
        conn.close()


def get_matching_score(conn: sqlite3.Connection, id_a: str, id_b: str) -> float | None:
    num_a, num_b = _id_to_num(id_a), _id_to_num(id_b)
    lo, hi = (num_a, num_b) if num_a < num_b else (num_b, num_a)
    row = conn.execute(
        "SELECT score FROM matching_scores WHERE drug_a_num = ? AND drug_b_num = ?",
        (lo, hi),
    ).fetchone()
    return row["score"] if row else None


# ---------------------------------------------------------------------------
# Similarity-based replacement suggestions
# ---------------------------------------------------------------------------

def find_similar_replacements(
    conn: sqlite3.Connection,
    drug_id: str,
    regime_ids: set[str],
    cutoff: float = SIMILARITY_CUTOFF,
) -> list[dict]:
    drug_num = _id_to_num(drug_id)
    rows = conn.execute(
        """
        SELECT 'DDInter' || ms.drug_b_num AS cand_id, ms.score
        FROM matching_scores ms
        WHERE ms.drug_a_num = ? AND ms.score >= ?
        UNION ALL
        SELECT 'DDInter' || ms.drug_a_num AS cand_id, ms.score
        FROM matching_scores ms
        WHERE ms.drug_b_num = ? AND ms.score >= ?
        """,
        (drug_num, cutoff, drug_num, cutoff),
    ).fetchall()

    results = []
    for row in rows:
        cand_id = row[0]
        if cand_id in regime_ids:
            continue
        cand_num = _id_to_num(cand_id)
        info_row = conn.execute(
            """
            SELECT
                COUNT(*) AS cnt,
                COALESCE(
                    MAX(CASE WHEN drug_a_num = ? THEN drug_a_name END),
                    MAX(CASE WHEN drug_b_num = ? THEN drug_b_name END)
                ) AS name
            FROM interactions
            WHERE drug_a_num = ? OR drug_b_num = ?
            """,
            (cand_num, cand_num, cand_num, cand_num),
        ).fetchone()
        results.append({
            "id":                cand_id,
            "name":              info_row["name"] if info_row else cand_id,
            "score":             row[1],
            "interaction_count": info_row["cnt"]  if info_row else 0,
        })

    results.sort(key=lambda r: r["interaction_count"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Risk calculation helpers
# ---------------------------------------------------------------------------

def drug_avg_strength(conn: sqlite3.Connection, drug_id: str, regime_ids: list[str] | None = None) -> float | None:
    if regime_ids is not None:
        other_ids = [rid for rid in regime_ids if rid != drug_id]
        if not other_ids:
            return None
        drug_num   = _id_to_num(drug_id)
        other_nums = [_id_to_num(rid) for rid in other_ids]
        placeholders = ",".join("?" * len(other_nums))
        row = conn.execute(
            f"""
            SELECT AVG(strength) AS avg_s
            FROM interactions
            WHERE (drug_a_num = ? AND drug_b_num IN ({placeholders}))
               OR (drug_b_num = ? AND drug_a_num IN ({placeholders}))
            """,
            [drug_num] + other_nums + [drug_num] + other_nums,
        ).fetchone()
    else:
        drug_num = _id_to_num(drug_id)
        row = conn.execute(
            """
            SELECT AVG(strength) AS avg_s
            FROM interactions
            WHERE drug_a_num = ? OR drug_b_num = ?
            """,
            (drug_num, drug_num),
        ).fetchone()
    return row["avg_s"]


def blended_drug_risk(
    interaction_strength: float | None,
    se_burden: float | None,
    se_weight: float = SE_WEIGHT,
) -> float:
    """
    Blend interaction strength (0–3 scale) with side-effect burden (0–1 scale,
    normalised to 0–3) using se_weight.  Missing values default to 0.
    """
    interaction = interaction_strength if interaction_strength is not None else 0.0
    # Normalise SE burden from [0,1] to [0,3] to match interaction scale
    se = (se_burden * 3.0) if se_burden is not None else 0.0
    return (1.0 - se_weight) * interaction + se_weight * se


def calc_normalized_risk(conn: sqlite3.Connection, drug_ids: list[str]) -> float | None:
    if not drug_ids:
        return None
    risks = []
    for did in drug_ids:
        strength = drug_avg_strength(conn, did, regime_ids=drug_ids)
        burden   = drug_se_burden(conn, did)
        risks.append(blended_drug_risk(strength, burden))
    return sum(risks) / len(risks)


def pair_has_interaction(conn: sqlite3.Connection, id_a: str, id_b: str) -> bool:
    num_a, num_b = _id_to_num(id_a), _id_to_num(id_b)
    row = conn.execute(
        """
        SELECT 1 FROM interactions
        WHERE (drug_a_num = ? AND drug_b_num = ?)
           OR (drug_a_num = ? AND drug_b_num = ?)
        LIMIT 1
        """,
        (num_a, num_b, num_b, num_a),
    ).fetchone()
    return row is not None


def resolve_drug(conn: sqlite3.Connection, query: str) -> dict | None:
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
            FROM interactions WHERE drug_a_num = ? LIMIT 1
            """,
            (num,),
        ).fetchone()
        if not row:
            row = conn.execute(
                """
                SELECT 'DDInter' || drug_b_num AS id, drug_b_name AS name
                FROM interactions WHERE drug_b_num = ? LIMIT 1
                """,
                (num,),
            ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT 'DDInter' || drug_a_num AS id, drug_a_name AS name
            FROM interactions WHERE LOWER(drug_a_name) = LOWER(?) LIMIT 1
            """,
            (query,),
        ).fetchone()
        if not row:
            row = conn.execute(
                """
                SELECT 'DDInter' || drug_b_num AS id, drug_b_name AS name
                FROM interactions WHERE LOWER(drug_b_name) = LOWER(?) LIMIT 1
                """,
                (query,),
            ).fetchone()
    return dict(row) if row else None


def search_drugs(conn: sqlite3.Connection, query: str, limit: int = 10) -> list[dict]:
    pattern = f"%{query}%"
    rows = conn.execute(
        """
        SELECT DISTINCT 'DDInter' || drug_a_num AS id, drug_a_name AS name
        FROM interactions
        WHERE LOWER(drug_a_name) LIKE LOWER(?)
        UNION
        SELECT DISTINCT 'DDInter' || drug_b_num AS id, drug_b_name AS name
        FROM interactions
        WHERE LOWER(drug_b_name) LIKE LOWER(?)
        LIMIT ?
        """,
        (pattern, pattern, limit),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Drug-food interaction loading
# ---------------------------------------------------------------------------

_FOOD_SCHEMA = """
CREATE TABLE IF NOT EXISTS food_interactions (
    drug_num    INTEGER NOT NULL,
    food_name   TEXT    NOT NULL,
    severity    INTEGER NOT NULL,
    description TEXT    NOT NULL,
    management  TEXT    NOT NULL,
    mechanism   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_food_drug ON food_interactions(drug_num);
"""

_FOOD_INSERT_SQL = """
INSERT INTO food_interactions (drug_num, food_name, severity, description, management, mechanism)
VALUES (?, ?, ?, ?, ?, ?)
"""


def build_food_database(csv_path: str, db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript("DROP TABLE IF EXISTS food_interactions;" + _FOOD_SCHEMA)
        conn.commit()

        if not Path(csv_path).exists():
            print(f"[warn] Food CSV not found at '{csv_path}' — food interactions disabled.", file=sys.stderr)
            return

        cur = conn.cursor()
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            batch = []
            skipped = 0
            for row in reader:
                sev_raw = (row.get("Severity level") or "").strip()
                try:
                    severity = int(sev_raw)
                except ValueError:
                    skipped += 1
                    continue
                drug_id_str = (row.get("drug_id") or "").strip()
                if not drug_id_str or not drug_id_str.upper().startswith("DDINTER"):
                    skipped += 1
                    continue
                try:
                    drug_num = _id_to_num(drug_id_str)
                except (ValueError, IndexError):
                    skipped += 1
                    continue
                batch.append((
                    drug_num,
                    (row.get("Food name") or "").strip(),
                    severity,
                    (row.get("Description") or "").strip(),
                    (row.get("Management") or "").strip(),
                    (row.get("Mechanism") or "").strip(),
                ))
            if batch:
                cur.executemany(_FOOD_INSERT_SQL, batch)

        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM food_interactions").fetchone()[0]
        print(f"[info] Loaded {count:,} drug-food interactions ({skipped:,} rows skipped).")
    finally:
        conn.close()


def get_food_interactions(conn: sqlite3.Connection, drug_id: str) -> list[dict]:
    drug_num = _id_to_num(drug_id)
    rows = conn.execute(
        """
        SELECT food_name, severity, description, management, mechanism
        FROM food_interactions
        WHERE drug_num = ?
        ORDER BY severity DESC, food_name ASC
        """,
        (drug_num,),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Drug-disease interaction loading
# ---------------------------------------------------------------------------

_DISEASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS disease_interactions (
    drug_num     INTEGER NOT NULL,
    disease_name TEXT    NOT NULL,
    severity     INTEGER NOT NULL,
    text         TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_disease_drug ON disease_interactions(drug_num);
"""

_DISEASE_INSERT_SQL = """
INSERT INTO disease_interactions (drug_num, disease_name, severity, text)
VALUES (?, ?, ?, ?)
"""


def build_disease_database(csv_path: str, db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript("DROP TABLE IF EXISTS disease_interactions;" + _DISEASE_SCHEMA)
        conn.commit()

        if not Path(csv_path).exists():
            print(f"[warn] Disease CSV not found at '{csv_path}' — disease interactions disabled.", file=sys.stderr)
            return

        cur = conn.cursor()
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            batch = []
            skipped = 0
            for row in reader:
                sev_raw = (row.get("Severity level") or "").strip()
                try:
                    severity = int(sev_raw)
                except ValueError:
                    skipped += 1
                    continue
                drug_id_str = (row.get("drug_id") or "").strip()
                if not drug_id_str or not drug_id_str.upper().startswith("DDINTER"):
                    skipped += 1
                    continue
                try:
                    drug_num = _id_to_num(drug_id_str)
                except (ValueError, IndexError):
                    skipped += 1
                    continue
                batch.append((
                    drug_num,
                    (row.get("Disease name") or "").strip(),
                    severity,
                    (row.get("Text") or "").strip(),
                ))
            if batch:
                cur.executemany(_DISEASE_INSERT_SQL, batch)

        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM disease_interactions").fetchone()[0]
        print(f"[info] Loaded {count:,} drug-disease interactions ({skipped:,} rows skipped).")
    finally:
        conn.close()


def get_disease_interactions(conn: sqlite3.Connection, drug_id: str) -> list[dict]:
    drug_num = _id_to_num(drug_id)
    rows = conn.execute(
        """
        SELECT disease_name, severity, text
        FROM disease_interactions
        WHERE drug_num = ?
        ORDER BY severity DESC, disease_name ASC
        """,
        (drug_num,),
    ).fetchall()
    return [dict(r) for r in rows]


def drug_se_burden(conn: sqlite3.Connection, drug_id: str) -> float | None:
    """
    Returns the average freq_upper across all side effects for this drug (0–1 scale).
    Entries without a known frequency are excluded.  Returns None if no data exists.
    """
    drug_num = _id_to_num(drug_id)
    row = conn.execute(
        """
        SELECT AVG(freq_upper) AS burden
        FROM side_effects
        WHERE ddinter_num = ? AND freq_upper IS NOT NULL
        """,
        (drug_num,),
    ).fetchone()
    return row["burden"] if row and row["burden"] is not None else None


def get_side_effects(
    conn: sqlite3.Connection,
    drug_id: str,
    limit: int = 50,
) -> list[dict]:
    """Return side effects for a drug ordered by descending frequency upper bound."""
    drug_num = _id_to_num(drug_id)
    rows = conn.execute(
        """
        SELECT se_name, freq_lower, freq_upper, freq_label
        FROM side_effects
        WHERE ddinter_num = ?
          AND (freq_upper IS NULL OR freq_upper > 0)
          AND (freq_lower IS NULL OR freq_lower > 0)
        ORDER BY
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
        LIMIT ?
        """,
        (drug_num, limit),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    build_database(CSV_PATH, DB_PATH)
    build_matching_scores(DB_PATH)
    build_food_database(FOOD_CSV_PATH, DB_PATH)
    build_disease_database(DISEASE_CSV_PATH, DB_PATH)
    yield


app = FastAPI(title="Drug Interaction Risk Scorer", lifespan=lifespan)

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
    drug_ids: list[str]


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


class ReplacementCandidate(BaseModel):
    id:                str
    name:              str
    score:             float
    interaction_count: int
    substitute_risk:   float | None


class DrugReplacements(BaseModel):
    drug_id:                    str
    drug_name:                  str
    original_interaction_count: int
    replacements:               list[ReplacementCandidate]


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


class DrugSideEffects(BaseModel):
    drug_id:      str
    drug_name:    str
    side_effects: list[SideEffect]


class RegimeResponse(BaseModel):
    drugs:                list[DrugRisk]
    total_risk:           float
    normalized_risk:      float
    se_weight:            float
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

@app.get("/health")
def health():
    conn = get_conn()
    try:
        count = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
    finally:
        conn.close()
    return {"status": "ok", "interactions": count}


@app.get("/search")
def search(q: str, limit: int = 10):
    if len(q) < 2:
        return []
    conn = get_conn()
    try:
        return search_drugs(conn, q, limit)
    finally:
        conn.close()


@app.get("/drugs/{drug_id}/side_effects", response_model=DrugSideEffects)
def drug_side_effects(drug_id: str, limit: int = 50):
    conn = get_conn()
    try:
        drug = resolve_drug(conn, drug_id)
        if not drug:
            raise HTTPException(status_code=404, detail=f"Drug '{drug_id}' not found")
        effects = get_side_effects(conn, drug["id"], limit=limit)
        return DrugSideEffects(
            drug_id=drug["id"],
            drug_name=drug["name"],
            side_effects=[SideEffect(**e) for e in effects],
        )
    finally:
        conn.close()


@app.post("/regime/risk", response_model=RegimeResponse)
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
        drug_risks: list[DrugRisk] = []
        for drug in resolved:
            avg     = drug_avg_strength(conn, drug["id"], regime_ids=resolved_ids)
            burden  = drug_se_burden(conn, drug["id"])
            risk_val = blended_drug_risk(avg, burden)
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
        populated  = sum(1 for a, b in pairs if pair_has_interaction(conn, a, b))
        coverage   = (populated / possible * 100) if possible > 0 else 0.0

        pair_scores: list[PairScore] = []
        for id_a, id_b in pairs:
            score = get_matching_score(conn, id_a, id_b)
            na, nb = _id_to_num(id_a), _id_to_num(id_b)
            mech_row = conn.execute(
                """
                SELECT mechanism FROM interactions
                WHERE (drug_a_num = ? AND drug_b_num = ?)
                   OR (drug_a_num = ? AND drug_b_num = ?)
                LIMIT 1
                """,
                (na, nb, nb, na),
            ).fetchone()
            mechanism = mech_row["mechanism"] if mech_row else None
            pair_scores.append(PairScore(
                drug_a_id=id_a,
                drug_a_name=names[id_a],
                drug_b_id=id_b,
                drug_b_name=names[id_b],
                score=score,
                mechanism=mechanism,
            ))

        similar_replacements: list[DrugReplacements] = []
        for drug in resolved:
            candidates = find_similar_replacements(conn, drug["id"], regime_set)
            drug_num = _id_to_num(drug["id"])
            orig_count_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM interactions WHERE drug_a_num = ? OR drug_b_num = ?",
                (drug_num, drug_num),
            ).fetchone()
            orig_count = orig_count_row["cnt"] if orig_count_row else 0

            # For each candidate, compute regime risk with this drug swapped out
            substitute_regime = [rid for rid in resolved_ids if rid != drug["id"]]
            replacement_objects: list[ReplacementCandidate] = []
            for c in candidates:
                sub_ids = substitute_regime + [c["id"]]
                sub_risk = calc_normalized_risk(conn, sub_ids)
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

        food_interactions: list[DrugFoodInteractions] = []
        for drug in resolved:
            foods = get_food_interactions(conn, drug["id"])
            food_interactions.append(DrugFoodInteractions(
                drug_id=drug["id"],
                drug_name=drug["name"],
                foods=[FoodInteraction(**f) for f in foods],
            ))

        disease_interactions: list[DrugDiseaseInteractions] = []
        for drug in resolved:
            diseases = get_disease_interactions(conn, drug["id"])
            disease_interactions.append(DrugDiseaseInteractions(
                drug_id=drug["id"],
                drug_name=drug["name"],
                diseases=[DiseaseInteraction(**d) for d in diseases],
            ))

        side_effects: list[DrugSideEffects] = []
        for drug in resolved:
            effects = get_side_effects(conn, drug["id"])
            side_effects.append(DrugSideEffects(
                drug_id=drug["id"],
                drug_name=drug["name"],
                side_effects=[SideEffect(**e) for e in effects],
            ))

        return RegimeResponse(
            drugs=drug_risks,
            total_risk=total_risk,
            normalized_risk=round(normalized_risk, 6),
            se_weight=SE_WEIGHT,
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

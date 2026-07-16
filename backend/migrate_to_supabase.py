"""
One-time (re-runnable) import of all DDI and side-effect data into Supabase
Postgres, replacing the old "rebuild SQLite from CSV on every startup" model
in main.py. Run this whenever the source CSVs/TSVs change; the app itself no
longer rebuilds anything at request time.

Loads, in order:
  1. interactions.csv              -> interactions
  2. (computed from the above)     -> matching_scores  (Sorensen-Dice, O(n^2) over drugs)
  3. drug_food.csv                 -> food_interactions
  4. drug_disease.csv              -> disease_interactions
  5. cid_to_ddinter.csv + Drug_SE_DB/meddra_freq.tsv -> side_effects

Requires DATABASE_URL in backend/.env (Supabase connection string, transaction
pooler / port 6543 recommended). See backend/.env.example.

Usage:
    cd backend
    pip install -r requirements.txt
    python migrate_to_supabase.py
"""

import csv
import itertools
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).parent
ROOT_DIR = BACKEND_DIR.parent

load_dotenv(BACKEND_DIR / ".env")

DATABASE_URL = os.environ.get("DATABASE_URL")

CSV_PATH = BACKEND_DIR / os.environ.get("DRUG_CSV", "interactions.csv")
FOOD_CSV_PATH = BACKEND_DIR / os.environ.get("FOOD_CSV", "drug_food.csv")
DISEASE_CSV_PATH = BACKEND_DIR / os.environ.get("DISEASE_CSV", "drug_disease.csv")
MAPPING_CSV_PATH = BACKEND_DIR / "cid_to_ddinter.csv"
MEDDRA_FREQ_TSV = ROOT_DIR / "Drug_SE_DB" / "meddra_freq.tsv" / "meddra_freq.tsv"

SCHEMA_SQL_PATH = BACKEND_DIR / "schema.sql"


def _id_to_num(drug_id: str) -> int:
    """Strip the 'DDInter' prefix and return the integer drug number."""
    return int(drug_id[7:])


def _is_data_row(row: list[str]) -> bool:
    try:
        float(row[4])
        return True
    except (ValueError, IndexError):
        return False


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def apply_schema(conn: psycopg.Connection) -> None:
    print("[schema] Applying schema.sql...")
    sql = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    print("[schema] Done.")


# ---------------------------------------------------------------------------
# interactions
# ---------------------------------------------------------------------------

def load_interactions(conn: psycopg.Connection, csv_path: Path) -> None:
    if not csv_path.exists():
        print(f"[interactions] WARNING: {csv_path} not found, skipping.", file=sys.stderr)
        return

    print(f"[interactions] Loading from {csv_path}...")
    with conn.cursor() as cur:
        cur.execute("TRUNCATE interactions RESTART IDENTITY")

        count = 0
        with open(csv_path, newline="", encoding="utf-8") as f, \
             cur.copy(
                 "COPY interactions (drug_a_num, drug_a_name, drug_b_num, drug_b_name, strength, mechanism) "
                 "FROM STDIN"
             ) as copy:
            reader = csv.reader(f)
            first = next(reader, None)
            if first and not _is_data_row(first):
                rows = reader
            else:
                rows = itertools.chain([first], reader) if first else reader

            for row in rows:
                if len(row) < 5:
                    continue
                drug_a_id, drug_a_name, drug_b_id, drug_b_name, strength = row[:5]
                try:
                    strength_f = float(strength)
                except ValueError:
                    continue

                raw_mech = row[5].strip() if len(row) > 5 else ""
                mechanism = None
                if raw_mech and raw_mech.lower() != "unknown":
                    try:
                        float(raw_mech)  # numeric-only -> malformed row
                    except ValueError:
                        mechanism = raw_mech

                copy.write_row((
                    _id_to_num(drug_a_id.strip()), drug_a_name.strip(),
                    _id_to_num(drug_b_id.strip()), drug_b_name.strip(),
                    strength_f, mechanism,
                ))
                count += 1
                if count % 50_000 == 0:
                    print(f"  ...{count:,} rows")

    conn.commit()
    print(f"[interactions] Loaded {count:,} rows.")


# ---------------------------------------------------------------------------
# matching_scores (Sorensen-Dice over shared interaction neighbourhoods)
# ---------------------------------------------------------------------------

def build_matching_scores(conn: psycopg.Connection) -> None:
    print("[matching_scores] Computing Sorensen-Dice scores...")
    with conn.cursor() as cur:
        cur.execute("TRUNCATE matching_scores")

        rows = conn.execute(
            """
            SELECT drug_a_num AS id, drug_b_num AS neighbor FROM interactions
            UNION ALL
            SELECT drug_b_num AS id, drug_a_num AS neighbor FROM interactions
            """
        ).fetchall()

        neighbors: dict[int, set[int]] = {}
        for drug_num, neighbor_num in rows:
            neighbors.setdefault(drug_num, set()).add(neighbor_num)

        drug_nums = list(neighbors.keys())
        total_pairs = len(drug_nums) * (len(drug_nums) - 1) // 2
        print(f"[matching_scores] {len(drug_nums):,} drugs -> {total_pairs:,} pairs.")

        count = 0
        with cur.copy("COPY matching_scores (drug_a_num, drug_b_num, score) FROM STDIN") as copy:
            for num_a, num_b in itertools.combinations(drug_nums, 2):
                na, nb = neighbors[num_a], neighbors[num_b]
                denom = len(na) + len(nb)
                score = (2 * len(na & nb) / denom) if denom > 0 else 0.0
                lo, hi = (num_a, num_b) if num_a < num_b else (num_b, num_a)
                copy.write_row((lo, hi, score))
                count += 1
                if count % 200_000 == 0:
                    print(f"  ...{count:,} pairs")

    conn.commit()
    print(f"[matching_scores] Inserted {count:,} rows.")


# ---------------------------------------------------------------------------
# food_interactions
# ---------------------------------------------------------------------------

def load_food_interactions(conn: psycopg.Connection, csv_path: Path) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE food_interactions RESTART IDENTITY")

    if not csv_path.exists():
        print(f"[food_interactions] WARNING: {csv_path} not found, skipping.", file=sys.stderr)
        return

    print(f"[food_interactions] Loading from {csv_path}...")
    count = 0
    skipped = 0
    with conn.cursor() as cur, \
         open(csv_path, newline="", encoding="utf-8") as f, \
         cur.copy(
             "COPY food_interactions (drug_num, food_name, severity, description, management, mechanism) "
             "FROM STDIN"
         ) as copy:
        for row in csv.DictReader(f):
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
            copy.write_row((
                drug_num,
                (row.get("Food name") or "").strip(),
                severity,
                (row.get("Description") or "").strip(),
                (row.get("Management") or "").strip(),
                (row.get("Mechanism") or "").strip(),
            ))
            count += 1

    conn.commit()
    print(f"[food_interactions] Loaded {count:,} rows ({skipped:,} skipped).")


# ---------------------------------------------------------------------------
# disease_interactions
# ---------------------------------------------------------------------------

def load_disease_interactions(conn: psycopg.Connection, csv_path: Path) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE disease_interactions RESTART IDENTITY")

    if not csv_path.exists():
        print(f"[disease_interactions] WARNING: {csv_path} not found, skipping.", file=sys.stderr)
        return

    print(f"[disease_interactions] Loading from {csv_path}...")
    count = 0
    skipped = 0
    with conn.cursor() as cur, \
         open(csv_path, newline="", encoding="utf-8") as f, \
         cur.copy(
             "COPY disease_interactions (drug_num, disease_name, severity, text) FROM STDIN"
         ) as copy:
        for row in csv.DictReader(f):
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
            copy.write_row((
                drug_num,
                (row.get("Disease name") or "").strip(),
                severity,
                (row.get("Text") or "").strip(),
            ))
            count += 1

    conn.commit()
    print(f"[disease_interactions] Loaded {count:,} rows ({skipped:,} skipped).")


# ---------------------------------------------------------------------------
# side_effects  (cid_to_ddinter.csv mapping + SIDER meddra_freq.tsv)
# ---------------------------------------------------------------------------

def load_mapping(path: Path) -> dict[str, tuple[str, int]]:
    """Returns {cid: (ddinter_name, ddinter_num)} for exact matches only."""
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["match_type"] == "exact":
                mapping[row["cid"]] = (row["ddinter_name"], int(row["ddinter_num"]))
    return mapping


def load_side_effects(conn: psycopg.Connection, mapping_csv: Path, meddra_tsv: Path) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE side_effects RESTART IDENTITY")

    if not mapping_csv.exists():
        print(f"[side_effects] WARNING: {mapping_csv} not found, skipping.", file=sys.stderr)
        return
    if not meddra_tsv.exists():
        print(f"[side_effects] WARNING: {meddra_tsv} not found, skipping.", file=sys.stderr)
        print("  (Download SIDER's meddra_freq.tsv into Drug_SE_DB/meddra_freq.tsv/ first.)", file=sys.stderr)
        return

    print("[side_effects] Loading CID->DDInter mapping...")
    mapping = load_mapping(mapping_csv)
    print(f"[side_effects] {len(mapping):,} usable mappings.")

    print(f"[side_effects] Parsing {meddra_tsv}...")
    best: dict[tuple[int, str], tuple[float | None, float | None, str, str]] = {}
    rows_read = 0
    with open(meddra_tsv, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for parts in reader:
            if len(parts) < 10:
                continue
            flat_cid = parts[0].strip()
            placebo = parts[3].strip()
            freq_desc = parts[4].strip()
            freq_lower = parts[5].strip()
            freq_upper = parts[6].strip()
            meddra_type = parts[7].strip()
            se_name = parts[9].strip()

            if meddra_type != "PT" or placebo == "placebo":
                continue
            if flat_cid not in mapping:
                continue

            ddinter_name, ddinter_num = mapping[flat_cid]
            try:
                fl = float(freq_lower) if freq_lower else None
                fu = float(freq_upper) if freq_upper else None
            except ValueError:
                fl, fu = None, None

            key = (ddinter_num, se_name)
            existing = best.get(key)
            if existing is None or (fu is not None and fu > existing[1]):
                best[key] = (fl, fu, freq_desc, ddinter_name)

            rows_read += 1
            if rows_read % 200_000 == 0:
                print(f"  ...{rows_read:,} rows read")

    print(f"[side_effects] {rows_read:,} rows parsed, {len(best):,} unique (drug, side effect) pairs.")

    print("[side_effects] Inserting...")
    with conn.cursor() as cur, cur.copy(
        "COPY side_effects (ddinter_name, ddinter_num, se_name, freq_lower, freq_upper, freq_label) FROM STDIN"
    ) as copy:
        for (num, se), (fl, fu, label, ddinter_name) in best.items():
            copy.write_row((ddinter_name, num, se, fl, fu, label))

    conn.commit()
    print(f"[side_effects] Inserted {len(best):,} rows.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not DATABASE_URL:
        print(
            "ERROR: DATABASE_URL is not set. Copy backend/.env.example to backend/.env "
            "and fill in your Supabase connection string.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Connecting to {DATABASE_URL.split('@')[-1]}...")
    with psycopg.connect(DATABASE_URL) as conn:
        apply_schema(conn)
        load_interactions(conn, CSV_PATH)
        build_matching_scores(conn)
        load_food_interactions(conn, FOOD_CSV_PATH)
        load_disease_interactions(conn, DISEASE_CSV_PATH)
        load_side_effects(conn, MAPPING_CSV_PATH, MEDDRA_FREQ_TSV)

    print("\nMigration complete.")


if __name__ == "__main__":
    main()

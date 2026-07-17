"""
Imports drug side effect data from SIDER's meddra_freq.tsv into interactions.db.

Steps:
  1. Load the cleaned CID->DDInter mapping (exact matches only)
  2. Parse meddra_freq.tsv, filter to PT (preferred terms) and non-placebo rows
  3. For each drug, aggregate side effects: take the max frequency across
     duplicate entries, then store (ddinter_name, se_name, freq_lower, freq_upper,
     freq_description) in a new `side_effects` table
"""

import csv
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "interactions.db")
MAPPING_CSV = os.path.join(os.path.dirname(__file__), "cid_to_ddinter.csv")
SE_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "Drug_SE_DB")
MEDDRA_FREQ_TSV = os.path.join(SE_DB_DIR, "meddra_freq.tsv", "meddra_freq.tsv")


def load_mapping(path: str) -> dict[str, tuple[str, int]]:
    """Returns {cid: (ddinter_name, ddinter_num)} for exact matches only."""
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["match_type"] == "exact":
                mapping[row["cid"]] = (row["ddinter_name"], int(row["ddinter_num"]))
    return mapping


def create_table(conn: sqlite3.Connection):
    conn.execute("DROP TABLE IF EXISTS side_effects")
    conn.execute(
        """
        CREATE TABLE side_effects (
            id            INTEGER PRIMARY KEY,
            ddinter_name  TEXT    NOT NULL,
            ddinter_num   INTEGER NOT NULL,
            se_name       TEXT    NOT NULL,
            freq_lower    REAL,
            freq_upper    REAL,
            freq_label    TEXT
        )
        """
    )
    conn.execute("CREATE INDEX idx_se_num ON side_effects(ddinter_num)")
    conn.execute("CREATE INDEX idx_se_name ON side_effects(ddinter_name)")


def parse_and_insert(conn: sqlite3.Connection, mapping: dict):
    # Track best (highest upper-bound) frequency per (ddinter_num, se_name)
    # so duplicate entries from multiple trials are deduplicated.
    best: dict[tuple[int, str], tuple[float, float, str]] = {}

    print("Parsing meddra_freq.tsv...")
    rows_read = 0
    with open(MEDDRA_FREQ_TSV, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for parts in reader:
            if len(parts) < 10:
                continue
            # columns: flat_cid, stereo_cid, umls_label, placebo,
            #          freq_desc, freq_lower, freq_upper,
            #          meddra_type, umls_meddra_id, se_name
            flat_cid     = parts[0].strip()
            placebo      = parts[3].strip()
            freq_desc    = parts[4].strip()
            freq_lower   = parts[5].strip()
            freq_upper   = parts[6].strip()
            meddra_type  = parts[7].strip()
            se_name      = parts[9].strip()

            # Only preferred terms, skip placebo data
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
            # Keep the entry with the highest upper-bound frequency
            existing = best.get(key)
            if existing is None or (fu is not None and fu > existing[1]):
                best[key] = (fl, fu, freq_desc, ddinter_name)

            rows_read += 1
            if rows_read % 100_000 == 0:
                print(f"  ...{rows_read:,} rows read")

    print(f"Done. {rows_read:,} rows parsed, {len(best):,} unique (drug, side_effect) pairs.")

    print("Inserting into database...")
    rows = [
        (ddinter_name, num, se, fl, fu, label)
        for (num, se), (fl, fu, label, ddinter_name) in best.items()
    ]
    conn.executemany(
        "INSERT INTO side_effects (ddinter_name, ddinter_num, se_name, freq_lower, freq_upper, freq_label) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    print(f"Inserted {len(rows):,} rows into side_effects table.")


def main():
    print("Loading mapping...")
    mapping = load_mapping(MAPPING_CSV)
    print(f"  {len(mapping)} usable CID->DDInter mappings")

    conn = sqlite3.connect(DB_PATH)
    try:
        create_table(conn)
        parse_and_insert(conn, mapping)

        # Quick sanity check
        cur = conn.cursor()
        cur.execute("SELECT COUNT(DISTINCT ddinter_num) FROM side_effects")
        drug_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM side_effects")
        total = cur.fetchone()[0]
        cur.execute(
            "SELECT ddinter_name, se_name, freq_lower, freq_upper, freq_label "
            "FROM side_effects WHERE ddinter_name = 'Metformin' "
            "ORDER BY freq_upper DESC LIMIT 5"
        )
        sample = cur.fetchall()
        print(f"\nSanity check: {drug_count} drugs, {total:,} total side effect entries")
        print("Metformin top side effects:", sample)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

"""
Builds a mapping table (cid_to_ddinter.csv) between STITCH CID drug IDs
(from Drug_SE_DB) and DDInter drug names/IDs (from interactions.db).

Matching strategy (in order of confidence):
  1. Exact name match (case-insensitive)
  2. Fuzzy match using difflib SequenceMatcher (threshold configurable)

Output CSV columns: cid, cid_name, ddinter_name, match_type, score
"""

import csv
import sqlite3
import os
from difflib import SequenceMatcher, get_close_matches

# --- Paths ---
DB_PATH = os.path.join(os.path.dirname(__file__), "interactions.db")
SE_DB_DIR = r"C:\Users\andew\OneDrive\Documents\Drug_SE_DB"
DRUG_NAMES_TSV = os.path.join(SE_DB_DIR, "drug_names.tsv")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "cid_to_ddinter.csv")

FUZZY_THRESHOLD = 0.82  # tune this; lower = more matches but more false positives


def load_cid_names(path: str) -> dict[str, str]:
    """Returns {cid: drug_name} from drug_names.tsv."""
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                cid, name = parts[0].strip(), parts[1].strip()
                mapping[cid] = name
    return mapping


def load_ddinter_names(db_path: str) -> dict[str, int]:
    """Returns {lowercase_name: drug_num} from interactions table."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT drug_a_name, drug_a_num FROM interactions "
        "UNION SELECT DISTINCT drug_b_name, drug_b_num FROM interactions"
    )
    rows = cur.fetchall()
    conn.close()
    # keep first occurrence if duplicates
    names: dict[str, int] = {}
    for name, num in rows:
        key = name.lower().strip()
        if key not in names:
            names[key] = num
    return names


def fuzzy_score(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def build_mapping(
    cid_names: dict[str, str],
    ddinter_names: dict[str, int],
    threshold: float,
) -> list[dict]:
    ddinter_keys = list(ddinter_names.keys())
    results = []

    for cid, raw_name in cid_names.items():
        query = raw_name.lower().strip()

        # 1. Exact match
        if query in ddinter_names:
            results.append(
                {
                    "cid": cid,
                    "cid_name": raw_name,
                    "ddinter_name": raw_name,
                    "ddinter_num": ddinter_names[query],
                    "match_type": "exact",
                    "score": 1.0,
                }
            )
            continue

        # 2. Fuzzy match
        close = get_close_matches(query, ddinter_keys, n=1, cutoff=threshold)
        if close:
            best = close[0]
            score = fuzzy_score(query, best)
            results.append(
                {
                    "cid": cid,
                    "cid_name": raw_name,
                    "ddinter_name": best,
                    "ddinter_num": ddinter_names[best],
                    "match_type": "fuzzy",
                    "score": round(score, 4),
                }
            )
        else:
            results.append(
                {
                    "cid": cid,
                    "cid_name": raw_name,
                    "ddinter_name": None,
                    "ddinter_num": None,
                    "match_type": "unmatched",
                    "score": 0.0,
                }
            )

    return results


def main():
    print("Loading CID drug names...")
    cid_names = load_cid_names(DRUG_NAMES_TSV)
    print(f"  {len(cid_names)} CID entries loaded")

    print("Loading DDInter drug names from interactions.db...")
    ddinter_names = load_ddinter_names(DB_PATH)
    print(f"  {len(ddinter_names)} unique DDInter drug names loaded")

    print(f"Matching (fuzzy threshold={FUZZY_THRESHOLD})...")
    results = build_mapping(cid_names, ddinter_names, FUZZY_THRESHOLD)

    exact = sum(1 for r in results if r["match_type"] == "exact")
    fuzzy = sum(1 for r in results if r["match_type"] == "fuzzy")
    unmatched = sum(1 for r in results if r["match_type"] == "unmatched")

    print(f"\nResults: {exact} exact | {fuzzy} fuzzy | {unmatched} unmatched")
    print(f"Match rate: {(exact + fuzzy) / len(results) * 100:.1f}%")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["cid", "cid_name", "ddinter_name", "ddinter_num", "match_type", "score"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nMapping saved to: {OUTPUT_CSV}")

    # Print a sample of fuzzy matches for manual review
    print("\n--- Sample fuzzy matches (review for accuracy) ---")
    fuzzy_sample = [r for r in results if r["match_type"] == "fuzzy"][:20]
    for r in fuzzy_sample:
        print(f"  {r['cid_name']!r:30s} -> {r['ddinter_name']!r:30s}  (score={r['score']})")

    print("\n--- Unmatched CID drugs (first 20) ---")
    for r in [r for r in results if r["match_type"] == "unmatched"][:20]:
        print(f"  {r['cid']}  {r['cid_name']!r}")


if __name__ == "__main__":
    main()

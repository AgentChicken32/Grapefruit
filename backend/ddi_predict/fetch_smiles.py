"""
Step 1 of the DDI prediction pipeline: for every DDInter drug, resolve a
PubChem CID (via cid_to_ddinter.csv) where possible and fetch its
chemical/textual data:

  - canonical SMILES         (structure, embedded via ChemBERTa)
  - IUPAC name                 } (natural language, embedded via a
  - PubChem description        }  general text model)
  - ATC therapeutic category  (derived locally from interactions.csv —
                                the most common atc_category value across
                                that drug's own interaction rows; DDInter
                                doesn't give a clean per-drug ATC field, so
                                this is a best-effort majority vote)

SMILES/IUPAC/description are only fetchable for drugs with a resolved
PubChem CID (currently ~907 of 2,265 drugs) — the rest still get a row
with just name + ATC category, since embed_smiles.py can fall back to
name/ATC-only text for drugs with no chemical structure.

Results are cached to smiles_cache.csv. Re-runs are resumable per field —
a drug already holding a SMILES value won't be re-fetched for that field,
but will still be topped up if it's missing IUPAC name or description.

Usage:
    python fetch_smiles.py [--limit N] [--sleep 0.25]
"""
import argparse
import csv
import itertools
import time
from collections import Counter
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
BACKEND_DIR = HERE.parent
INTERACTIONS_CSV = BACKEND_DIR / "interactions.csv"
CID_MAP_CSV = BACKEND_DIR / "cid_to_ddinter.csv"
OUTPUT_CSV = HERE / "smiles_cache.csv"

PROPERTY_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES,IUPACName/JSON"
DESCRIPTION_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/description/JSON"

FIELDNAMES = ["ddinter_num", "ddinter_name", "atc_category", "cid", "smiles", "iupac_name", "description"]


def parse_pubchem_cid(stitch_cid: str) -> int | None:
    """
    STITCH/SIDER 'flat' CIDs look like 'CID100000119': 'CID' + a 1-digit
    stereo flag (0=flat, 1=stereo-specific) + a zero-padded PubChem CID.
    Strip both to recover the real numeric PubChem CID (119, here).
    """
    if not stitch_cid.upper().startswith("CID"):
        return None
    digits = stitch_cid[3:].strip()
    if not digits.isdigit():
        return None
    if len(digits) > 1:
        digits = digits[1:]  # drop the stereo/flat flag digit
    try:
        return int(digits)
    except ValueError:
        return None


def _is_data_row(row: list[str]) -> bool:
    try:
        float(row[4])
        return True
    except (ValueError, IndexError):
        return False


def load_ddinter_drugs(path: Path) -> dict[int, str]:
    """All unique {ddinter_num: name} pairs referenced in interactions.csv."""
    drugs: dict[int, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first and not _is_data_row(first):
            rows = reader
        else:
            rows = itertools.chain([first], reader) if first else reader
        for row in rows:
            if len(row) < 4:
                continue
            for id_col, name_col in ((0, 1), (2, 3)):
                raw_id = row[id_col].strip()
                if not raw_id.upper().startswith("DDINTER"):
                    continue
                try:
                    num = int(raw_id[7:])
                except ValueError:
                    continue
                drugs.setdefault(num, row[name_col].strip())
    return drugs


def load_atc_categories(path: Path) -> dict[int, str]:
    """
    Best-effort per-drug ATC category: DDInter's atc_category column
    describes the interaction row, not a specific drug, so we take the
    most common category across all rows a drug appears in (as either
    side of the pair) as its representative therapeutic category.
    """
    counters: dict[int, Counter] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first and not _is_data_row(first):
            rows = reader
        else:
            rows = itertools.chain([first], reader) if first else reader
        for row in rows:
            if len(row) < 7:
                continue
            category = row[6].strip()
            if not category:
                continue
            for id_col in (0, 2):
                raw_id = row[id_col].strip()
                if not raw_id.upper().startswith("DDINTER"):
                    continue
                try:
                    num = int(raw_id[7:])
                except ValueError:
                    continue
                counters.setdefault(num, Counter())[category] += 1
    return {num: counter.most_common(1)[0][0] for num, counter in counters.items()}


def load_best_cid_per_drug(path: Path) -> dict[int, str]:
    """Best STITCH CID per ddinter_num: prefer exact matches, else highest fuzzy score."""
    best: dict[int, tuple[str, str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            num_raw = row.get("ddinter_num")
            if not num_raw:
                continue
            try:
                num = int(num_raw)
            except ValueError:
                continue
            match_type = row.get("match_type", "")
            try:
                score = float(row.get("score") or 0)
            except ValueError:
                score = 0.0
            rank = (1 if match_type == "exact" else 0, score)
            prev = best.get(num)
            prev_rank = (1 if prev and prev[1] == "exact" else 0, prev[2] if prev else -1.0)
            if prev is None or rank > prev_rank:
                best[num] = (row["cid"], match_type, score)
    return {num: v[0] for num, v in best.items()}


def load_cache(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {int(r["ddinter_num"]): r for r in csv.DictReader(f)}


def _get_json(url: str, session: requests.Session, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=15)
        except requests.RequestException:
            time.sleep(1 + attempt)
            continue
        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError:
                return None
        if resp.status_code == 404:
            return None
        if resp.status_code == 429:
            time.sleep(2 ** attempt * 2)
            continue
        time.sleep(1 + attempt)
    return None


def fetch_properties(pubchem_cid: int, session: requests.Session) -> tuple[str | None, str | None]:
    """Returns (smiles, iupac_name)."""
    data = _get_json(PROPERTY_URL.format(cid=pubchem_cid), session)
    if not data:
        return None, None
    props = data.get("PropertyTable", {}).get("Properties", [])
    if not props:
        return None, None
    prop = props[0]
    return prop.get("CanonicalSMILES"), prop.get("IUPACName")


def fetch_description(pubchem_cid: int, session: requests.Session) -> str | None:
    data = _get_json(DESCRIPTION_URL.format(cid=pubchem_cid), session)
    if not data:
        return None
    for info in data.get("InformationList", {}).get("Information", []):
        desc = (info.get("Description") or "").strip()
        if desc:
            return desc
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N drugs needing any field (useful for a smoke test)")
    parser.add_argument("--sleep", type=float, default=0.25,
                         help="Delay between PubChem requests, in seconds")
    args = parser.parse_args()

    print(f"[info] Loading DDInter drugs from {INTERACTIONS_CSV}")
    drugs = load_ddinter_drugs(INTERACTIONS_CSV)
    print(f"[info] {len(drugs):,} unique DDInter drugs")

    print("[info] Computing per-drug ATC categories (local, majority vote)...")
    atc_by_num = load_atc_categories(INTERACTIONS_CSV)
    print(f"[info] {len(atc_by_num):,} drugs have an inferred ATC category")

    print(f"[info] Loading CID mapping from {CID_MAP_CSV}")
    cid_by_num = load_best_cid_per_drug(CID_MAP_CSV)
    print(f"[info] {len(cid_by_num):,} drugs have a candidate PubChem CID")

    cache = load_cache(OUTPUT_CSV)
    print(f"[info] {len(cache):,} drugs already cached in {OUTPUT_CSV.name}")

    # SMILES is no longer a hard requirement — every DDInter drug is a
    # candidate. Drugs without a PubChem CID match simply end up with no
    # smiles/iupac_name/description; embed_smiles.py falls back to
    # name+ATC-category-only text for those (see its module docstring).
    def needs_fetch(num: int) -> bool:
        row = cache.get(num)
        if row is None:
            return True
        if num not in cid_by_num:
            return False  # nothing fetchable without a CID; already recorded
        return (
            not (row.get("smiles") or "").strip()
            or not (row.get("iupac_name") or "").strip()
            or not (row.get("description") or "").strip()
        )

    needs_work = [num for num in drugs if needs_fetch(num)]
    if args.limit:
        needs_work = needs_work[: args.limit]
    print(f"[info] {len(needs_work):,} drugs need fetching (new, or missing fetchable fields)...")

    session = requests.Session()
    for i, num in enumerate(needs_work, 1):
        row = dict(cache.get(num, {}))
        row["ddinter_num"] = num
        row["ddinter_name"] = drugs[num]
        row["atc_category"] = atc_by_num.get(num, "")

        stitch_cid = cid_by_num.get(num)
        pubchem_cid = parse_pubchem_cid(stitch_cid) if stitch_cid else None
        row["cid"] = pubchem_cid or ""

        if pubchem_cid is not None:
            if not (row.get("smiles") or "").strip() or not (row.get("iupac_name") or "").strip():
                smiles, iupac = fetch_properties(pubchem_cid, session)
                if smiles:
                    row["smiles"] = smiles
                if iupac:
                    row["iupac_name"] = iupac
                time.sleep(args.sleep)

            if not (row.get("description") or "").strip():
                desc = fetch_description(pubchem_cid, session)
                if desc:
                    row["description"] = desc
                time.sleep(args.sleep)

        # Always cache — name (+ ATC category, when known) is enough
        # signal on its own for the text-embedding fallback.
        cache[num] = row

        if i % 50 == 0:
            print(f"[info] {i}/{len(needs_work)} processed")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for num in sorted(cache.keys()):
            row = cache[num]
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})

    with_smiles = sum(1 for r in cache.values() if (r.get("smiles") or "").strip())
    with_iupac = sum(1 for r in cache.values() if (r.get("iupac_name") or "").strip())
    with_desc = sum(1 for r in cache.values() if (r.get("description") or "").strip())
    print(f"[info] Done. {len(cache):,} drugs cached total "
          f"({with_smiles:,} with SMILES, {with_iupac:,} with IUPAC name, "
          f"{with_desc:,} with description) in {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

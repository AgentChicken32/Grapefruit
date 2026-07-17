"""
Scores each unique side effect name in the side_effects table with a severity
rating of 1 (mild), 2 (moderate), or 3 (severe), matching the interaction
strength scale already used for drug-drug interactions.

Uses Claude Opus 4.8 to batch-score 50 names per API call.
Resumable: skips side effects that already have a severity score.

Usage:
    python score_se_severity.py

Requirements:
    - ANTHROPIC_API_KEY environment variable set
    - anthropic package installed in the venv
"""

import json
import os
import sqlite3
import time

import anthropic

DB_PATH = os.path.join(os.path.dirname(__file__), "interactions.db")
BATCH_SIZE = 50
MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """You are a clinical pharmacology expert. You will be given a list of drug side effect
names and must rate each one for clinical severity on a 1–5 scale:

1 = Trivial — barely noticeable, no impact on daily life, no intervention needed
    (e.g., dry mouth, mild drowsiness, slight headache, minor GI upset)

2 = Mild — noticeable but manageable without medical attention, self-limiting
    (e.g., nausea, fatigue, dizziness, mild rash, insomnia, decreased appetite)

3 = Moderate — interferes with normal activity or requires medical attention /
    dose adjustment, but not life-threatening
    (e.g., vomiting, significant hypertension, elevated liver enzymes, moderate
    allergic reaction, depression, QT prolongation, significant weight gain)

4 = Severe — serious harm, hospitalisation likely, or significant long-term impact
    (e.g., severe hepatotoxicity, agranulocytosis, rhabdomyolysis, serious
    arrhythmia, major bleeding, severe anaphylaxis, pulmonary embolism)

5 = Life-threatening / Potentially fatal — immediate medical intervention required,
    high risk of death or permanent disability
    (e.g., fatal hepatic failure, Stevens-Johnson syndrome, toxic epidermal
    necrolysis, myocardial infarction, stroke, aplastic anaemia, sudden cardiac death)

Respond ONLY with a valid JSON object mapping each exact side effect name to its
score. Do not include any explanation or extra text.

Example response format:
{"dry mouth": 1, "nausea": 2, "agranulocytosis": 4, "fatal hepatic failure": 5}"""


def add_severity_column(conn: sqlite3.Connection):
    """Add severity column if it doesn't exist."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(side_effects)").fetchall()]
    if "severity" not in cols:
        conn.execute("ALTER TABLE side_effects ADD COLUMN severity INTEGER")
        conn.commit()
        print("Added 'severity' column to side_effects table.")


def get_unscored_names(conn: sqlite3.Connection) -> list[str]:
    """Return distinct SE names that have no severity score yet."""
    rows = conn.execute(
        "SELECT DISTINCT se_name FROM side_effects WHERE severity IS NULL ORDER BY se_name"
    ).fetchall()
    return [r[0] for r in rows]


def score_batch(client: anthropic.Anthropic, names: list[str]) -> dict[str, int]:
    """Ask Claude to score a batch of SE names. Returns {name: score}."""
    names_json = json.dumps(names)
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Score these side effect names:\n{names_json}",
            }
        ],
    )
    text = response.content[0].text.strip()
    scores = json.loads(text)
    # Validate that values are 1, 2, or 3
    result = {}
    for name in names:
        if name in scores and scores[name] in (1, 2, 3, 4, 5):
            result[name] = scores[name]
        else:
            print(f"  WARNING: Missing or invalid score for '{name}', defaulting to 3")
            result[name] = 3
    return result


def apply_scores(conn: sqlite3.Connection, scores: dict[str, int]):
    """Write severity scores to all matching rows."""
    conn.executemany(
        "UPDATE side_effects SET severity = ? WHERE se_name = ?",
        [(v, k) for k, v in scores.items()],
    )
    conn.commit()


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "ERROR: ANTHROPIC_API_KEY environment variable is not set.\n"
            "Set it before running this script."
        )

    client = anthropic.Anthropic(api_key=api_key)
    conn = sqlite3.connect(DB_PATH)

    try:
        add_severity_column(conn)

        names = get_unscored_names(conn)
        total = len(names)
        if total == 0:
            print("All side effects already have severity scores. Nothing to do.")
            return

        print(f"Scoring {total} unique side effect names in batches of {BATCH_SIZE}...")
        batches = [names[i : i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
        scored = 0

        for i, batch in enumerate(batches, 1):
            print(f"  Batch {i}/{len(batches)} ({len(batch)} names)...", end=" ", flush=True)
            try:
                scores = score_batch(client, batch)
                apply_scores(conn, scores)
                scored += len(scores)
                print(f"done ({scored}/{total} total)")
            except json.JSONDecodeError as e:
                print(f"PARSE ERROR (will retry next run): {e}")
            except anthropic.RateLimitError:
                print("RATE LIMITED — waiting 60s...")
                time.sleep(60)
                # Retry the same batch
                scores = score_batch(client, batch)
                apply_scores(conn, scores)
                scored += len(scores)
                print(f"done after retry ({scored}/{total} total)")
            # Small pause to be kind to rate limits
            time.sleep(0.5)

        # Summary
        count = conn.execute(
            "SELECT COUNT(*) FROM side_effects WHERE severity IS NOT NULL"
        ).fetchone()[0]
        dist = conn.execute(
            "SELECT severity, COUNT(*) FROM side_effects WHERE severity IS NOT NULL "
            "GROUP BY severity ORDER BY severity"
        ).fetchall()
        print(f"\nDone! {count} rows now have severity scores.")
        for sev, cnt in dist:
            label = {1: "trivial", 2: "mild", 3: "moderate", 4: "severe", 5: "life-threatening"}.get(sev, "?")
            print(f"  {sev} ({label}): {cnt} rows")

    finally:
        conn.close()


if __name__ == "__main__":
    main()

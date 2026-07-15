-- Postgres schema for Supabase, mirroring the SQLite schema previously
-- rebuilt from CSV on every backend startup (see main.py's _SCHEMA,
-- _MATCHING_SCHEMA, _FOOD_SCHEMA, _DISEASE_SCHEMA, and import_side_effects.py).
--
-- Run once via migrate_to_supabase.py (which executes this file), or
-- paste into the Supabase SQL editor.

CREATE TABLE IF NOT EXISTS interactions (
    id          BIGSERIAL PRIMARY KEY,
    drug_a_num  INTEGER NOT NULL,
    drug_a_name TEXT    NOT NULL,
    drug_b_num  INTEGER NOT NULL,
    drug_b_name TEXT    NOT NULL,
    strength    DOUBLE PRECISION NOT NULL,
    mechanism   TEXT
);
CREATE INDEX IF NOT EXISTS idx_interactions_a ON interactions(drug_a_num);
CREATE INDEX IF NOT EXISTS idx_interactions_b ON interactions(drug_b_num);

CREATE TABLE IF NOT EXISTS matching_scores (
    drug_a_num INTEGER NOT NULL,
    drug_b_num INTEGER NOT NULL,
    score      DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (drug_a_num, drug_b_num)
);

CREATE TABLE IF NOT EXISTS food_interactions (
    id          BIGSERIAL PRIMARY KEY,
    drug_num    INTEGER NOT NULL,
    food_name   TEXT    NOT NULL,
    severity    INTEGER NOT NULL,
    description TEXT    NOT NULL,
    management  TEXT    NOT NULL,
    mechanism   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_food_drug ON food_interactions(drug_num);

CREATE TABLE IF NOT EXISTS disease_interactions (
    id           BIGSERIAL PRIMARY KEY,
    drug_num     INTEGER NOT NULL,
    disease_name TEXT    NOT NULL,
    severity     INTEGER NOT NULL,
    text         TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_disease_drug ON disease_interactions(drug_num);

CREATE TABLE IF NOT EXISTS side_effects (
    id           BIGSERIAL PRIMARY KEY,
    ddinter_name TEXT    NOT NULL,
    ddinter_num  INTEGER NOT NULL,
    se_name      TEXT    NOT NULL,
    freq_lower   DOUBLE PRECISION,
    freq_upper   DOUBLE PRECISION,
    freq_label   TEXT
);
CREATE INDEX IF NOT EXISTS idx_se_num  ON side_effects(ddinter_num);
CREATE INDEX IF NOT EXISTS idx_se_name ON side_effects(ddinter_name);

-- This data is a public, read-only reference dataset with no per-user rows,
-- so RLS is left disabled and all tables are readable via the anon key.
-- (Re-enable and add SELECT-only policies later if that changes.)
ALTER TABLE interactions          DISABLE ROW LEVEL SECURITY;
ALTER TABLE matching_scores       DISABLE ROW LEVEL SECURITY;
ALTER TABLE food_interactions     DISABLE ROW LEVEL SECURITY;
ALTER TABLE disease_interactions  DISABLE ROW LEVEL SECURITY;
ALTER TABLE side_effects          DISABLE ROW LEVEL SECURITY;

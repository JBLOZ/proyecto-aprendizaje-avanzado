-- Canonical SQLite schema for the LexGLUE / ECtHR Task B project.
--
-- The schema is intentionally normalized because the project has three
-- different analytical needs:
--   1. document-level modelling (`cases`);
--   2. paragraph-level auditability (`case_paragraphs`);
--   3. multilabel supervision (`case_labels` + `articles`);
--   4. experiment traceability (`experiment_runs`, `predictions`,
--      `explanations`).
--
-- Running this file from an empty SQLite database is sufficient to create the
-- full structure used by all notebooks:
--
--   sqlite3 data/interim/metadata.db ".read schema.sql"
--
-- The notebooks call the same file through `sqlite3.executescript`, so the
-- database definition has a single source of truth.

PRAGMA foreign_keys = ON;

DROP VIEW IF EXISTS v_case_label_summary;
DROP VIEW IF EXISTS v_case_label_codes;

DROP TABLE IF EXISTS explanations;
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS experiment_runs;
DROP TABLE IF EXISTS case_labels;
DROP TABLE IF EXISTS case_paragraphs;
DROP TABLE IF EXISTS articles;
DROP TABLE IF EXISTS cases;

CREATE TABLE cases (
    -- Stable synthetic id: <task>_<split>_<row_number>.
    case_id TEXT PRIMARY KEY,

    -- Dataset task, currently `ecthr_task_b`.
    task TEXT NOT NULL,

    -- Official benchmark split. The evaluation protocol depends on this
    -- column: train fits models, validation tunes thresholds, test is final.
    split TEXT NOT NULL CHECK (split IN ('train', 'validation', 'test')),

    -- Official year when available. LexGLUE ECtHR Task B does not expose this
    -- metadata in the Hugging Face records used here, so values may be NULL.
    year INTEGER CHECK (year IS NULL OR (year BETWEEN 1950 AND 2035)),

    -- Full case text obtained by joining the original fact paragraphs.
    text_full TEXT NOT NULL,

    -- Basic document length metadata used in EDA and drift checks.
    n_paragraphs INTEGER NOT NULL CHECK (n_paragraphs >= 0),
    n_tokens INTEGER NOT NULL CHECK (n_tokens >= 0)
);

CREATE TABLE case_paragraphs (
    case_id TEXT NOT NULL,
    paragraph_idx INTEGER NOT NULL CHECK (paragraph_idx >= 0),
    paragraph_text TEXT NOT NULL,
    PRIMARY KEY (case_id, paragraph_idx),
    FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE
);

CREATE TABLE articles (
    -- LexGLUE stores labels as integer ids. We keep that id as text-compatible
    -- primary key because downstream JSON artefacts also use string keys.
    article_id TEXT PRIMARY KEY,

    -- Human-facing ECHR article code, e.g. `6` or `P1-1`.
    article_code TEXT NOT NULL UNIQUE,

    -- Short explanation used in notebooks and report tables.
    description TEXT NOT NULL
);

CREATE TABLE case_labels (
    case_id TEXT NOT NULL,
    article_id TEXT NOT NULL,
    value INTEGER NOT NULL DEFAULT 1 CHECK (value IN (0, 1)),
    PRIMARY KEY (case_id, article_id),
    FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE,
    FOREIGN KEY (article_id) REFERENCES articles(article_id) ON DELETE CASCADE
);

CREATE TABLE experiment_runs (
    run_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    model_name TEXT NOT NULL,
    config_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    git_commit TEXT
);

CREATE TABLE predictions (
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    y_true_json TEXT NOT NULL,
    y_pred_json TEXT NOT NULL,
    scores_json TEXT,
    PRIMARY KEY (run_id, case_id),
    FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE
);

CREATE TABLE explanations (
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    method TEXT NOT NULL,
    artifact_path TEXT,
    summary_json TEXT NOT NULL,
    PRIMARY KEY (run_id, case_id, method),
    FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE
);

CREATE INDEX idx_cases_split ON cases(split);
CREATE INDEX idx_cases_year ON cases(year);
CREATE INDEX idx_case_labels_article ON case_labels(article_id);
CREATE INDEX idx_case_labels_case ON case_labels(case_id);
CREATE INDEX idx_predictions_case ON predictions(case_id);

CREATE VIEW v_case_label_codes AS
SELECT
    c.case_id,
    c.split,
    c.year,
    c.n_paragraphs,
    c.n_tokens,
    a.article_code,
    a.description
FROM cases AS c
JOIN case_labels AS cl ON cl.case_id = c.case_id
JOIN articles AS a ON a.article_id = cl.article_id;

CREATE VIEW v_case_label_summary AS
SELECT
    c.case_id,
    c.split,
    c.year,
    c.n_paragraphs,
    c.n_tokens,
    COUNT(cl.article_id) AS n_positive_articles,
    GROUP_CONCAT(a.article_code, ', ') AS article_codes
FROM cases AS c
LEFT JOIN case_labels AS cl ON cl.case_id = c.case_id
LEFT JOIN articles AS a ON a.article_id = cl.article_id
GROUP BY c.case_id, c.split, c.year, c.n_paragraphs, c.n_tokens;

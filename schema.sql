DROP TABLE IF EXISTS explanations;
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS experiment_runs;
DROP TABLE IF EXISTS case_labels;
DROP TABLE IF EXISTS case_paragraphs;
DROP TABLE IF EXISTS articles;
DROP TABLE IF EXISTS cases;

CREATE TABLE cases (
    case_id TEXT PRIMARY KEY,
    task TEXT,
    split TEXT,
    year INTEGER,
    text_full TEXT,
    n_paragraphs INTEGER,
    n_tokens INTEGER
);

CREATE TABLE case_paragraphs (
    case_id TEXT,
    paragraph_idx INTEGER,
    paragraph_text TEXT,
    PRIMARY KEY (case_id, paragraph_idx)
);

CREATE TABLE articles (
    article_id TEXT PRIMARY KEY,
    article_code TEXT,
    description TEXT
);

CREATE TABLE case_labels (
    case_id TEXT,
    article_id TEXT,
    value INTEGER,
    PRIMARY KEY (case_id, article_id)
);

CREATE TABLE experiment_runs (
    run_id TEXT PRIMARY KEY,
    stage TEXT,
    model_name TEXT,
    config_json TEXT,
    metrics_json TEXT,
    created_at TEXT,
    git_commit TEXT
);

CREATE TABLE predictions (
    run_id TEXT,
    case_id TEXT,
    y_true_json TEXT,
    y_pred_json TEXT,
    scores_json TEXT,
    PRIMARY KEY (run_id, case_id)
);

CREATE TABLE explanations (
    run_id TEXT,
    case_id TEXT,
    method TEXT,
    artifact_path TEXT,
    summary_json TEXT,
    PRIMARY KEY (run_id, case_id, method)
);

CREATE INDEX idx_cases_split ON cases(split);
CREATE INDEX idx_cases_year ON cases(year);
CREATE INDEX idx_case_labels_article ON case_labels(article_id);

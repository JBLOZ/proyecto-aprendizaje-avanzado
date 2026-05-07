"""Utilidades compartidas para los notebooks multietiqueta de ECtHR.

Este archivo contiene solo codigo reutilizado por varios notebooks: rutas del
proyecto, ingesta/carga de SQLite, construccion de la matriz multietiqueta,
metricas comunes y pequenos helpers matematicos. El entrenamiento de modelos,
retrieval, XAI y graficas especificas viven en sus notebooks correspondientes.
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd


SEED = 42
TASK = "ecthr_task_b"
DATASET_NAME = "lex_glue"
DATASET_SUBSET = "ecthr_b"

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
GENERATED_FIGURES = FIGURES / "generated"
METRICS = ARTIFACTS / "metrics"
MODELS = ARTIFACTS / "models"
REPORTS = ARTIFACTS / "reports"
INDICES = ARTIFACTS / "indices"
DB = INTERIM / "metadata.db"
SCHEMA = ROOT / "schema.sql"

SPLIT_ORDER = ["train", "validation", "test"]
ARTICLE_CODES = ["2", "3", "5", "6", "8", "9", "10", "11", "14", "P1-1"]
ARTICLE_DESCRIPTIONS = {
    "2": "Derecho a la vida",
    "3": "Prohibicion de tortura o tratos inhumanos/degradantes",
    "5": "Derecho a la libertad y seguridad",
    "6": "Derecho a un proceso equitativo",
    "8": "Derecho al respeto de la vida privada y familiar",
    "9": "Libertad de pensamiento, conciencia y religion",
    "10": "Libertad de expresion",
    "11": "Libertad de reunion y asociacion",
    "14": "Prohibicion de discriminacion",
    "P1-1": "Proteccion de la propiedad",
}


def configure(seed: int = SEED) -> None:
    """Crea carpetas del proyecto y fija semillas de Python y NumPy.

    Entrada:
        seed: semilla entera usada para reproducibilidad.

    Salida:
        No devuelve nada. Crea directorios si faltan y deja inicializadas las
        semillas globales usadas por los notebooks.
    """

    random.seed(seed)
    np.random.seed(seed)
    for folder in [RAW, INTERIM, ARTIFACTS, FIGURES, GENERATED_FIGURES, METRICS, MODELS, REPORTS, INDICES]:
        folder.mkdir(parents=True, exist_ok=True)


def git_commit() -> str | None:
    """Devuelve el hash del commit actual de Git.

    Entrada:
        Ninguna.

    Salida:
        Cadena con el hash de `HEAD` si el repositorio esta disponible; `None`
        si el comando Git no puede ejecutarse.
    """

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def connect_db() -> sqlite3.Connection:
    """Abre la base SQLite del proyecto con claves foraneas activadas.

    Entrada:
        Ninguna.

    Salida:
        Conexion `sqlite3.Connection` configurada con `row_factory` para poder
        leer filas por nombre de columna.
    """

    configure()
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Recrea la estructura SQLite a partir de `schema.sql`."""

    configure()
    with connect_db() as conn:
        conn.executescript(SCHEMA.read_text(encoding="utf-8"))
        conn.commit()


def normalize_paragraphs(record: dict) -> list[str]:
    """Extrae y limpia los parrafos de un registro de LexGLUE."""

    value = record.get("text") or record.get("facts") or record.get("document") or ""
    if isinstance(value, str):
        paragraphs = [value]
    elif isinstance(value, Iterable):
        paragraphs = []
        for item in value:
            if isinstance(item, str):
                paragraphs.append(item)
            elif isinstance(item, Iterable):
                paragraphs.append(" ".join(str(x) for x in item))
            else:
                paragraphs.append(str(item))
    else:
        paragraphs = [str(value)]
    return [p.strip() for p in paragraphs if str(p).strip()]


def extract_year_from_record(record: dict) -> int | None:
    """Extrae el anio oficial si el dataset lo proporciona."""

    for key in ["year", "judgment_year", "decision_year"]:
        value = record.get(key)
        if isinstance(value, (int, np.integer)) and 1950 <= int(value) <= 2035:
            return int(value)
    return None


def load_lexglue_dataset():
    """Carga LexGLUE ECtHR Task B usando el cache local del repositorio."""

    from datasets import load_dataset

    configure()
    return load_dataset(DATASET_NAME, DATASET_SUBSET, cache_dir=str(RAW / "hf_cache"))




def database_ready() -> bool:
    """Comprueba si la base canonica ya contiene las tablas principales.

    Entrada:
        Ninguna.

    Salida:
        `True` si existen casos, articulos y etiquetas; `False` si la base no
        existe, esta vacia o no tiene el formato esperado.
    """

    if not DB.exists():
        return False
    try:
        with connect_db() as conn:
            n_cases = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
            n_articles = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
            n_labels = conn.execute("SELECT COUNT(*) FROM case_labels").fetchone()[0]
        return n_cases > 0 and n_articles == len(ARTICLE_CODES) and n_labels > 0
    except sqlite3.Error:
        return False




def materialize_database(force: bool = False) -> dict:
    """Construye `metadata.db` desde LexGLUE y `schema.sql`.

    Entrada:
        force: si es `True`, borra y recrea las tablas aunque la base exista.

    Salida:
        Diccionario de estado con numero de casos, parrafos, articulos,
        etiquetas positivas y ruta de la base creada.
    """

    configure()
    if database_ready() and not force:
        return database_status()

    init_db()
    dataset = load_lexglue_dataset()
    article_rows = [(str(i), code, ARTICLE_DESCRIPTIONS[code]) for i, code in enumerate(ARTICLE_CODES)]

    with connect_db() as conn:
        conn.executemany(
            "INSERT INTO articles(article_id, article_code, description) VALUES (?, ?, ?)",
            article_rows,
        )
        for split in SPLIT_ORDER:
            for idx, record in enumerate(dataset[split]):
                case_id = f"{TASK}_{split}_{idx:06d}"
                paragraphs = normalize_paragraphs(record)
                text_full = "\n\n".join(paragraphs)
                tokens = re.findall(r"\b\w+\b", text_full)
                conn.execute(
                    """
                    INSERT INTO cases(case_id, task, split, year, text_full, n_paragraphs, n_tokens)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (case_id, TASK, split, extract_year_from_record(record), text_full, len(paragraphs), len(tokens)),
                )
                conn.executemany(
                    """
                    INSERT INTO case_paragraphs(case_id, paragraph_idx, paragraph_text)
                    VALUES (?, ?, ?)
                    """,
                    [(case_id, p_idx, p) for p_idx, p in enumerate(paragraphs)],
                )
                label_values = record.get("labels") or record.get("label") or []
                if isinstance(label_values, (int, np.integer)):
                    label_values = [int(label_values)]
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO case_labels(case_id, article_id, value)
                    VALUES (?, ?, 1)
                    """,
                    [(case_id, str(int(label))) for label in label_values],
                )
        conn.commit()

    status = database_status()
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "ingestion_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return status


def database_status() -> dict:
    """Resume el estado de la base canonica.

    Entrada:
        Ninguna.

    Salida:
        Diccionario con conteos globales y por split. Sirve para comprobar que
        la ingesta se completo correctamente.
    """

    with connect_db() as conn:
        split_counts = pd.read_sql_query(
            "SELECT split, COUNT(*) AS n_cases FROM cases GROUP BY split ORDER BY split",
            conn,
        ).to_dict(orient="records")
        return {
            "n_cases": conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0],
            "n_paragraphs": conn.execute("SELECT COUNT(*) FROM case_paragraphs").fetchone()[0],
            "n_articles": conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0],
            "n_positive_case_article_pairs": conn.execute("SELECT COUNT(*) FROM case_labels").fetchone()[0],
            "split_counts": split_counts,
            "db_path": str(DB),
            "schema_path": str(SCHEMA),
        }


def load_cases_labels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga casos, etiquetas positivas y metadatos de articulos.

    Entrada:
        Ninguna. Lee las tablas SQLite ya materializadas.

    Salida:
        Tupla `(cases, labels, articles)`:
        - `cases`: una fila por caso con texto y metadatos.
        - `labels`: pares positivos caso-articulo.
        - `articles`: catalogo de articulos y descripciones.
    """

    materialize_database(force=False)
    with connect_db() as conn:
        cases = pd.read_sql_query(
            "SELECT case_id, task, split, year, text_full, n_paragraphs, n_tokens FROM cases ORDER BY case_id",
            conn,
        )
        labels = pd.read_sql_query(
            "SELECT case_id, article_id, value FROM case_labels ORDER BY case_id, article_id",
            conn,
        )
        articles = pd.read_sql_query(
            "SELECT article_id, article_code, description FROM articles ORDER BY CAST(article_id AS INTEGER)",
            conn,
        )
    labels["article_id"] = labels["article_id"].astype(str)
    articles["article_id"] = articles["article_id"].astype(str)
    return cases, labels, articles


def multilabel_matrix(labels: pd.DataFrame, case_ids: Iterable[str], articles: pd.DataFrame) -> pd.DataFrame:
    """Construye la matriz binaria caso-articulo.

    Entrada:
        labels: pares positivos reales `case_id`-`article_id`.
        case_ids: orden de casos que formara el indice de filas.
        articles: catalogo de articulos que define las columnas.

    Salida:
        DataFrame `Y` con una fila por caso y una columna por articulo. Un `1`
        indica que el articulo es positivo para ese caso.
    """

    case_index = pd.Index(list(case_ids), name="case_id")
    article_ids = list(articles["article_id"].astype(str))
    y = pd.DataFrame(0, index=case_index, columns=article_ids, dtype=int)
    for row in labels[labels["case_id"].isin(case_index)].itertuples(index=False):
        y.loc[row.case_id, str(row.article_id)] = int(row.value)
    y.columns = list(articles["article_code"])
    return y


def train_validation_test_frames() -> dict[str, pd.DataFrame]:
    """Devuelve los DataFrames finales separados por split.

    Entrada:
        Ninguna.

    Salida:
        Diccionario con claves `train`, `validation` y `test`. Cada DataFrame
        contiene texto completo, metadatos y columnas binarias de articulos.
    """

    cases, labels, articles = load_cases_labels()
    y = multilabel_matrix(labels, cases["case_id"], articles)
    data = cases.join(y, on="case_id")
    return {split: data[data["split"] == split].reset_index(drop=True) for split in SPLIT_ORDER}


def summarize_case(case_id: str, max_chars: int = 900) -> dict:
    """Resume un caso real para mostrarlo en notebooks.

    Entrada:
        case_id: identificador del caso en SQLite.
        max_chars: numero maximo de caracteres del extracto textual.

    Salida:
        Diccionario con metadatos, articulos reales y extracto legible.
    """

    with connect_db() as conn:
        case = pd.read_sql_query("SELECT * FROM cases WHERE case_id = ?", conn, params=[case_id]).iloc[0].to_dict()
        article_codes = pd.read_sql_query(
            """
            SELECT a.article_code
            FROM case_labels cl
            JOIN articles a ON a.article_id = cl.article_id
            WHERE cl.case_id = ?
            ORDER BY CAST(a.article_id AS INTEGER)
            """,
            conn,
            params=[case_id],
        )["article_code"].tolist()
    text = case["text_full"].replace("\n", " ")
    case["excerpt"] = text[:max_chars] + ("..." if len(text) > max_chars else "")
    case["article_codes"] = ", ".join(article_codes)
    case.pop("text_full", None)
    return case


def example_cases() -> dict[str, pd.DataFrame]:
    """Devuelve conjuntos de ejemplos reales.
    Entrada:
        Ninguna.

    Salida:
        Diccionario de DataFrames con casos de muchas etiquetas, casos largos,
        ejemplos del articulo 9 y ejemplos del articulo P1-1.
    """

    materialize_database(force=False)
    queries = {
        "most_labels": """
            SELECT c.case_id, c.split, c.year, c.n_tokens,
                   COUNT(cl.article_id) AS n_labels,
                   GROUP_CONCAT(a.article_code, ', ') AS article_codes,
                   SUBSTR(REPLACE(c.text_full, x'0A', ' '), 1, 750) AS excerpt
            FROM cases c
            JOIN case_labels cl ON cl.case_id = c.case_id
            JOIN articles a ON a.article_id = cl.article_id
            GROUP BY c.case_id
            ORDER BY n_labels DESC, c.n_tokens DESC
            LIMIT 8
        """,
        "longest": """
            SELECT case_id, split, year, n_paragraphs, n_tokens,
                   SUBSTR(REPLACE(text_full, x'0A', ' '), 1, 750) AS excerpt
            FROM cases
            ORDER BY n_tokens DESC
            LIMIT 8
        """,
        "rare_article_9": """
            SELECT c.case_id, c.split, c.year, c.n_tokens, a.article_code,
                   SUBSTR(REPLACE(c.text_full, x'0A', ' '), 1, 750) AS excerpt
            FROM cases c
            JOIN case_labels cl ON cl.case_id = c.case_id
            JOIN articles a ON a.article_id = cl.article_id
            WHERE a.article_code = '9'
            ORDER BY c.n_tokens DESC
            LIMIT 5
        """,
        "property_article": """
            SELECT c.case_id, c.split, c.year, c.n_tokens, a.article_code,
                   SUBSTR(REPLACE(c.text_full, x'0A', ' '), 1, 750) AS excerpt
            FROM cases c
            JOIN case_labels cl ON cl.case_id = c.case_id
            JOIN articles a ON a.article_id = cl.article_id
            WHERE a.article_code = 'P1-1'
            ORDER BY c.n_tokens DESC
            LIMIT 5
        """,
    }
    with connect_db() as conn:
        return {name: pd.read_sql_query(sql, conn) for name, sql in queries.items()}


def metric_table(y_true: np.ndarray, y_pred: np.ndarray, split: str, model: str) -> dict:
    """Calcula las metricas multietiqueta usadas en el proyecto.

    Entrada:
        y_true: matriz binaria real `(n_casos, n_articulos)`.
        y_pred: matriz binaria predicha con la misma forma.
        split: nombre del split evaluado.
        model: nombre del modelo evaluado.

    Salida:
        Diccionario con macro-F1, micro-F1, Hamming loss y numero de casos.
    """

    from sklearn.metrics import f1_score, hamming_loss

    return {
        "model": model,
        "split": split,
        "n_cases": int(y_true.shape[0]),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }

def as_probability_matrix(raw) -> np.ndarray:
    """Normaliza salidas de probabilidad de sklearn.

    Entrada:
        raw: salida de `predict_proba`, que puede ser matriz o lista de
        matrices en algunos clasificadores multietiqueta.

    Salida:
        Matriz NumPy `(n_casos, n_articulos)` con la probabilidad/scores de la
        clase positiva para cada articulo.
    """

    if isinstance(raw, list):
        columns = []
        for item in raw:
            arr = np.asarray(item)
            columns.append(arr[:, 1] if arr.ndim == 2 and arr.shape[1] > 1 else arr.ravel())
        return np.vstack(columns).T
    return np.asarray(raw)

def load_predictions(run_id: str = "notebook_threshold_tuned") -> pd.DataFrame:
    """Carga predicciones guardadas en SQLite y decodifica sus JSON.

    Entrada:
        run_id: identificador de la ejecucion a recuperar.

    Salida:
        DataFrame con `case_id`, `split`, `y_true`, `y_pred` y `scores` ya
        convertidos a listas numericas.
    """

    materialize_database(force=False)
    with connect_db() as conn:
        pred = pd.read_sql_query(
            """
            SELECT p.run_id, p.case_id, c.split, p.y_true_json, p.y_pred_json, p.scores_json
            FROM predictions p
            JOIN cases c ON c.case_id = p.case_id
            WHERE p.run_id = ?
            ORDER BY p.case_id
            """,
            conn,
            params=[run_id],
        )
    if pred.empty:
        return pred
    pred["y_true"] = pred["y_true_json"].map(json.loads)
    pred["y_pred"] = pred["y_pred_json"].map(json.loads)
    pred["scores"] = pred["scores_json"].map(lambda x: json.loads(x) if isinstance(x, str) else None)
    return pred


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Calcula divergencia Jensen-Shannon entre distribuciones discretas.

    Entrada:
        p, q: vectores no necesariamente normalizados.

    Salida:
        Divergencia Jensen-Shannon en base 2. Valores mayores indican mas
        diferencia distribucional.
    """

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum() if p.sum() else np.ones_like(p) / len(p)
    q = q / q.sum() if q.sum() else np.ones_like(q) / len(q)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

configure()

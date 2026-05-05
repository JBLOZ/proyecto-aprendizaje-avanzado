# Sistema de apoyo al análisis inicial de casos ante el TEDH

## Recuperación de precedentes y predicción explicable de artículos jurídicos con análisis de shift entre particiones

Este repositorio contiene un proyecto de aprendizaje automático aplicado al derecho sobre **LexGLUE / ECtHR Task B**. Dado un caso jurídico descrito por sus hechos, el sistema sugiere artículos candidatos del Convenio Europeo de Derechos Humanos, recupera precedentes similares y genera señales de explicación y auditoría.

El sistema **no predice sentencias**, no determina responsabilidad jurídica y no sustituye a jueces, abogados ni investigadores. El encuadre correcto del proyecto es:

> Sistema de apoyo al análisis inicial de casos ante el TEDH: recuperación de precedentes y predicción explicable de artículos jurídicos con análisis de shift entre particiones.

---

## 1. Motivación

El análisis inicial de un caso ante el Tribunal Europeo de Derechos Humanos exige identificar artículos potencialmente implicados, revisar jurisprudencia previa y priorizar líneas argumentales. Es una tarea intensiva en lectura, sensible a omisiones y expuesta a sesgos de búsqueda.

El proyecto plantea un sistema de ML clásico, reproducible y auditable que ayuda a:

1. estructurar casos jurídicos largos en SQLite;
2. modelar ECtHR Task B como clasificación multietiqueta;
3. sugerir artículos candidatos mediante TF-IDF + One-vs-Rest;
4. recuperar precedentes similares sobre el conjunto de entrenamiento;
5. auditar señales globales/locales del modelo;
6. analizar drift temporal y patrones de error.

El foco metodológico está en **explicabilidad/XAI** y **shift entre particiones**. La privacidad se trata como consideración de despliegue responsable, no como eje experimental principal.

---

## 2. Dataset

Se usa **LexGLUE / ECtHR Task B**, un benchmark de NLP jurídico basado en casos del Tribunal Europeo de Derechos Humanos. La tarea consiste en identificar artículos del Convenio asociados a los hechos del caso.

| Elemento | Valor |
|---|---:|
| Train | 9000 casos |
| Validation | 1000 casos |
| Test | 1000 casos |
| Total | 11000 casos |
| Artículos / etiquetas | 10 |
| Pares positivos caso-artículo | 15991 |
| Media de artículos por caso etiquetado | 1.477 |
| Máximo de artículos en un caso | 7 |
| Párrafos almacenados | 262580 |

Etiquetas usadas:

```text
2, 3, 5, 6, 8, 9, 10, 11, 14, P1-1
```

El problema es multietiqueta, desbalanceado y de texto largo. Hay coocurrencias entre artículos, lo que hace insuficiente una formulación multiclase simple.

---

## 3. Problema formal

Cada caso se representa como un documento textual \(x_i\). La salida es un vector binario:

\[
y_i \in \{0,1\}^{K}, \quad K = 10
\]

donde \(y_{ij}=1\) indica que el artículo \(j\) está asociado al caso \(i\). El modelo aprende:

\[
f(x_i) \rightarrow s_i \in [0,1]^K
\]

y la predicción final aplica umbrales por etiqueta:

\[
\hat{y}_{ij} = \mathbb{1}(s_{ij} \geq \tau_j)
\]

Los umbrales \(\tau_j\) se ajustan en `validation`; `test` queda reservado para evaluación final.

---

## 4. Problemáticas identificadas

**Explicabilidad.** En derecho no basta con una etiqueta candidata: el usuario necesita auditar qué señales textuales empujan la predicción. Se usan coeficientes globales del modelo lineal y explicaciones locales LIME. Estas señales no son causalidad jurídica ni reemplazan argumentación doctrinal.

**Shift entre particiones.** La distribución de artículos y el rendimiento pueden cambiar entre `train`, `validation` y `test`. La versión de LexGLUE usada no expone un año oficial de sentencia, así que el proyecto no afirma deriva temporal estricta; mide divergencia Jensen-Shannon, distancia L1, similitud coseno y deltas de rendimiento respecto a validation.

**Errores.** Los falsos negativos pueden ocultar líneas argumentales relevantes; los falsos positivos pueden añadir carga de revisión y sesgar la investigación. El análisis de errores se presenta por split y patrón FP/FN.

**Privacidad.** El dataset usado es público, pero un despliegue real con expedientes sensibles requeriría anonimización, control de acceso, trazabilidad y, probablemente, ejecución local o en infraestructura aprobada.

---

## 5. Esquema SQLite

La ingesta materializa el dataset en `data/interim/metadata.db`.

```text
cases(
  case_id,
  task,
  split,
  year,
  text_full,
  n_paragraphs,
  n_tokens
)

case_paragraphs(
  case_id,
  paragraph_idx,
  paragraph_text
)

articles(
  article_id,
  article_code,
  description
)

case_labels(
  case_id,
  article_id,
  value
)

experiment_runs(
  run_id,
  stage,
  model_name,
  config_json,
  metrics_json,
  created_at,
  git_commit
)

predictions(
  run_id,
  case_id,
  y_true_json,
  y_pred_json,
  scores_json
)

explanations(
  run_id,
  case_id,
  method,
  artifact_path,
  summary_json
)
```

---

## 6. Estructura de notebooks

En la estructura real de esta entrega los notebooks están en la raíz del repositorio.

| Notebook | Función |
|---|---|
| `00_diagnostico_y_decisiones.ipynb` | Ingesta, normalización, SQLite y figuras de overview. |
| `01_datos_y_eda.ipynb` | EDA, desbalanceo, longitud, coocurrencias y Figura 3. |
| `02_modelado_multilabel.ipynb` | TF-IDF, Logistic Regression OVR, SVM OVR, red neuronal SVD+MLP, umbrales y Figura 4. |
| `03_retrieval_de_precedentes.ipynb` | Índices TF-IDF/SVD, evaluación sampled retrieval y Figura 5. |
| `04_xai_drift_y_errores.ipynb` | XAI global/local, shift entre particiones, errores y Figuras 6-7. |
| `05_guion_informe_y_defensa.ipynb` | Comprobación de artefactos, mapa de evidencias y guion de defensa. |

Pipeline final:

```text
00_diagnostico_y_decisiones.ipynb
01_datos_y_eda.ipynb
02_modelado_multilabel.ipynb
03_retrieval_de_precedentes.ipynb
04_xai_drift_y_errores.ipynb
05_guion_informe_y_defensa.ipynb
```

---

## 7. Instalación

Crear y activar un entorno:

```bash
python -m venv .venv
```

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Dependencias principales:

```text
numpy, pandas, scipy, scikit-learn, matplotlib, joblib,
datasets, lime, seaborn, nbconvert, nbformat, ipykernel
```

---

## 8. Ejecución reproducible

Ejecutar desde la raíz del repositorio:

```bash
jupyter nbconvert --to notebook --execute 00_diagnostico_y_decisiones.ipynb --inplace
jupyter nbconvert --to notebook --execute 01_datos_y_eda.ipynb --inplace
jupyter nbconvert --to notebook --execute 02_modelado_multilabel.ipynb --inplace
jupyter nbconvert --to notebook --execute 03_retrieval_de_precedentes.ipynb --inplace
jupyter nbconvert --to notebook --execute 04_xai_drift_y_errores.ipynb --inplace
jupyter nbconvert --to notebook --execute 05_guion_informe_y_defensa.ipynb --inplace
```

El notebook 02 reutiliza artefactos de modelo si existen y, en cualquier caso, regenera las predicciones en SQLite. Si faltan artefactos, entrena TF-IDF + Logistic Regression OVR, Linear SVM OVR y una comparación neuronal ligera TF-IDF + SVD + MLP de forma determinista con `SEED = 42`.

---

## 9. Resultados principales

### Clasificación multietiqueta

| Modelo | Split | Casos | Macro-F1 | Micro-F1 | Hamming loss |
|---|---:|---:|---:|---:|---:|
| TF-IDF + LogReg OVR + umbrales | train | 9000 | 0.877 | 0.905 | 0.028 |
| TF-IDF + LogReg OVR + umbrales | validation | 1000 | 0.740 | 0.774 | 0.062 |
| TF-IDF + LogReg OVR + umbrales | test | 1000 | 0.714 | 0.752 | 0.069 |
| TF-IDF + SVD + MLP + umbrales | validation | 1000 | 0.736 | 0.782 | 0.061 |
| TF-IDF + SVD + MLP + umbrales | test | 1000 | 0.729 | 0.770 | 0.065 |

La tabla completa está en `artifacts/metrics/paper_classification_table.csv`. También se genera `classification_model_comparison.csv` con baseline, Logistic 0.5, SVM lineal, Logistic con umbrales ajustados y MLP ligero.

### Retrieval de precedentes

| Método | Split | Consultas | nDCG@10 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|---:|
| TF-IDF coseno sampled | validation | 250 | 0.882 | 0.952 | 0.964 |
| TF-IDF coseno sampled | test | 250 | 0.867 | 0.944 | 0.972 |
| TF-IDF + SVD coseno sampled | validation | 250 | 0.899 | 0.964 | 0.964 |
| TF-IDF + SVD coseno sampled | test | 250 | 0.877 | 0.940 | 0.956 |

Advertencia: retrieval se evalúa sobre una muestra reproducible de 250 consultas por split y un índice construido sobre 2500 casos de train.

### XAI

| n etiquetas globales | n explicaciones locales | peso global medio absoluto | probabilidad local positiva media |
|---:|---:|---:|---:|
| 10 | 5 | 1.832 | 0.787 |

### Shift entre particiones

| Split objetivo | JS divergence | L1 | Cosine similarity | Δ macro-F1 | Δ micro-F1 | Δ Hamming |
|---|---:|---:|---:|---:|---:|---:|
| validation | 0.017 | 0.278 | 0.959 | 0.000 | 0.000 | 0.000 |
| test | 0.026 | 0.320 | 0.951 | -0.026 | -0.023 | +0.007 |

### Patrones de error en test

| FP | FN | Casos |
|---:|---:|---:|
| 0 | 0 | 463 |
| 1 | 0 | 194 |
| 0 | 1 | 165 |
| 1 | 1 | 82 |
| 0 | 2 | 35 |
| 2 | 0 | 30 |

Promedios de error:

| Split | TP | FP | FN | Errores |
|---|---:|---:|---:|---:|
| train | 1.358 | 0.180 | 0.105 | 0.285 |
| validation | 1.057 | 0.282 | 0.334 | 0.616 |
| test | 1.045 | 0.300 | 0.390 | 0.690 |

---

## 10. Artefactos

Figuras científicas:

```text
artifacts/figures/fig01_system_overview.pdf/.png
artifacts/figures/fig02_state_transition_pipeline.pdf/.png
artifacts/figures/fig03_eda_multipanel.pdf/.png
artifacts/figures/fig04_modeling_and_thresholds.pdf/.png
artifacts/figures/fig05_retrieval_precedents.pdf/.png
artifacts/figures/fig06_xai_global_local.pdf/.png
artifacts/figures/fig07_drift_error_distributions.pdf/.png
artifacts/figures/fig08_contributions_summary.pdf/.png
```

Tablas finales:

```text
artifacts/metrics/paper_classification_table.csv
artifacts/metrics/paper_retrieval_table.csv
artifacts/metrics/paper_xai_table.csv
artifacts/metrics/paper_drift_table.csv
artifacts/metrics/paper_error_pattern_table.csv
artifacts/metrics/paper_error_split_table.csv
```

Modelos e índices:

```text
artifacts/models/notebook_tfidf_vectorizer.joblib
artifacts/models/notebook_logreg_ovr.joblib
artifacts/models/notebook_svm_ovr.joblib
artifacts/models/notebook_svd_mlp.joblib
artifacts/models/notebook_thresholds.json
artifacts/indices/notebook_retrieval_tfidf_svd_index.joblib
```

Ilustraciones conceptuales generadas:

```text
artifacts/figures/generated/generated_01_system_flow.png
artifacts/figures/generated/generated_02_state_transition.png
artifacts/figures/generated/generated_03_false_positive_negative.png
artifacts/figures/generated/generated_04_temporal_drift.png
artifacts/figures/generated/generated_05_visual_abstract.png
artifacts/figures/generated/notebook_00_ingestion_schema.png
artifacts/figures/generated/notebook_01_eda.png
artifacts/figures/generated/notebook_02_modeling.png
artifacts/figures/generated/notebook_03_retrieval.png
artifacts/figures/generated/notebook_04_audit.png
artifacts/figures/generated/notebook_05_evidence.png
artifacts/figures/generated/imagegen_prompts.md
```

---

## 11. Reproducibilidad

El proyecto usa semilla fija:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
```

Decisiones de reproducibilidad:

- TF-IDF se ajusta solo sobre `train`.
- Los umbrales se ajustan con `validation`.
- `test` se reserva para evaluación final.
- Retrieval usa muestras reproducibles: 2500 casos de train y 250 queries por split.
- Las métricas finales proceden de CSV en `artifacts/metrics/`.
- Las figuras científicas se regeneran desde celdas incluidas en los notebooks correspondientes, sin depender de scripts externos.

---

## 12. Limitaciones

- TF-IDF no modela semántica contextual profunda ni relaciones jurídicas complejas.
- LIME y coeficientes lineales indican señales textuales, no causalidad jurídica.
- La relevancia de retrieval se aproxima por artículos compartidos, no por evaluación experta de precedentes.
- Los falsos negativos pueden ocultar artículos relevantes.
- Los falsos positivos pueden añadir carga de revisión y anclar la investigación.
- El drift observado puede reflejar evolución jurisprudencial, cambios de litigación o diferencias de muestreo.
- El sistema necesita supervisión humana obligatoria.

---

## 13. Trabajo futuro

- Evaluación cualitativa de precedentes por juristas.
- Ventanas temporales más finas y políticas explícitas de reentrenamiento.
- Calibración probabilística y análisis de umbrales por coste de error.
- Embeddings jurídicos o modelos largos como comparación controlada, no como sustituto de la auditoría.
- Evaluaciones de fidelidad de explicaciones más extensas.
- Integración de privacidad, anonimización y control de acceso para despliegues reales.

---

## 14. Uso de IA generativa

Se utilizó asistencia de IA generativa para revisar estructura, mejorar visualizaciones, redactar documentación, completar el informe LaTeX y generar ilustraciones conceptuales suplementarias. Las figuras científicas principales se generan con Matplotlib desde datos y métricas reproducibles dentro de los notebooks. Los prompts de las ilustraciones conceptuales quedan documentados en `artifacts/figures/generated/imagegen_prompts.md` y en el anexo de la memoria.

Las ilustraciones generadas no sustituyen resultados experimentales. Solo se usan como material conceptual o suplementario cuando su calidad es suficiente y no contienen afirmaciones métricas no verificadas.

---

## 15. Frase de alcance

> Este trabajo no busca automatizar la decisión judicial, sino proporcionar una herramienta de apoyo para priorizar artículos candidatos y precedentes relevantes en el análisis inicial de casos ante el TEDH.

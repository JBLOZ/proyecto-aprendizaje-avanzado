# Sistema de apoyo al analisis inicial de casos ante el TEDH

Este proyecto aplica aprendizaje automatico a **LexGLUE ECtHR Task B**. El sistema recibe los hechos de un caso y ayuda a revisar posibles articulos del Convenio Europeo de Derechos Humanos.

El sistema no predice sentencias. No decide si existe una violacion juridica. Sirve como apoyo para una revision humana inicial.

## 1. Objetivo

El proyecto construye un flujo reproducible para tres tareas.

1. Sugerir articulos candidatos del Convenio.
2. Recuperar precedentes similares del conjunto de entrenamiento.
3. Revisar senales de explicabilidad, shift entre particiones y errores.

El enfoque principal es clasico y auditable. Usa SQLite, TF-IDF, modelos One-vs-Rest, `GridSearchCV`, validacion cruzada de 5 folds y metricas multietiqueta.

## 2. Dataset

Se usa **LexGLUE ECtHR Task B**. La tarea consiste en identificar articulos del Convenio asociados a los hechos de cada caso.

| Elemento | Valor |
|---|---:|
| Train | 9000 casos |
| Validation | 1000 casos |
| Test | 1000 casos |
| Total | 11000 casos |
| Articulos | 10 |
| Pares positivos caso articulo | 15991 |
| Media de articulos por caso etiquetado | 1.477 |
| Maximo de articulos en un caso | 7 |
| Parrafos almacenados | 262580 |

Etiquetas usadas

```text
2, 3, 5, 6, 8, 9, 10, 11, 14, P1-1
```

El problema es multietiqueta. Un caso puede tener mas de un articulo positivo.

## 3. Protocolo experimental

El protocolo sigue el criterio final de evaluacion indicado por el profesor.

1. `train` se usa para seleccionar hiperparametros con `GridSearchCV`.
2. La validacion cruzada usa 5 folds.
3. TF-IDF esta dentro de un `Pipeline`.
4. SVD, escalado y MLP tambien estan dentro del `Pipeline` cuando se usan.
5. La metrica de seleccion es macro-F1.
6. El mejor estimador se reentrena con todo `train`.
7. `test` se usa una sola vez para la evaluacion final.
8. `test` no se usa para ajustar hiperparametros, umbrales ni arquitectura.

`validation` existe como particion oficial del benchmark y se usa para analisis auxiliares. No decide el modelo final ni los hiperparametros principales.

## 4. Notebooks

Los notebooks estan en la raiz del repositorio.

| Notebook | Funcion |
|---|---|
| `00_ingesta_y_esquema_relacional.ipynb` | Carga LexGLUE y crea la base SQLite. |
| `01_EDA.ipynb` | Analiza splits, longitud, desbalanceo y coocurrencias. |
| `02_modelado_multilabel.ipynb` | Define pipelines, ejecuta `GridSearchCV`, evalua test y guarda modelos. |
| `03_retrieval_de_precedentes.ipynb` | Recupera precedentes con similitud coseno. |
| `04_xai_drift_y_errores.ipynb` | Revisa explicabilidad, shift entre particiones y errores. |
| `05_guion_informe_y_defensa.ipynb` | Consolida evidencias y mensajes de defensa. |

Orden de ejecucion

```text
00_ingesta_y_esquema_relacional.ipynb
01_EDA.ipynb
02_modelado_multilabel.ipynb
03_retrieval_de_precedentes.ipynb
04_xai_drift_y_errores.ipynb
05_guion_informe_y_defensa.ipynb
```

## 5. Instalacion

Crear entorno

```bash
python -m venv .venv
```

Activar entorno en Windows

```bash
.venv\Scripts\activate
```

Activar entorno en macOS o Linux

```bash
source .venv/bin/activate
```

Instalar dependencias

```bash
pip install -r requirements.txt
```

## 6. Ejecucion reproducible

Ejecutar desde la raiz del repositorio.

```bash
jupyter nbconvert --to notebook --execute 00_ingesta_y_esquema_relacional.ipynb --inplace
jupyter nbconvert --to notebook --execute 01_EDA.ipynb --inplace
jupyter nbconvert --to notebook --execute 02_modelado_multilabel.ipynb --inplace
jupyter nbconvert --to notebook --execute 03_retrieval_de_precedentes.ipynb --inplace
jupyter nbconvert --to notebook --execute 04_xai_drift_y_errores.ipynb --inplace
jupyter nbconvert --to notebook --execute 05_guion_informe_y_defensa.ipynb --inplace
```

El notebook 02 puede tardar mas que el resto porque ejecuta las busquedas de hiperparametros. Los modelos finales se guardan como pipelines completos.

## 7. Resultados de clasificacion

Resultados finales en `test`.

| Modelo | Macro-F1 | Micro-F1 | Hamming loss |
|---|---:|---:|---:|
| SVM lineal OVR | 0.749 | 0.777 | 0.061 |
| LogReg OVR | 0.736 | 0.760 | 0.069 |
| SVD + MLP | 0.690 | 0.767 | 0.062 |
| Baseline frecuente | 0.057 | 0.324 | 0.165 |

Resultados de 5-fold CV sobre `train`.

| Modelo | Mejor macro-F1 CV | Desv. |
|---|---:|---:|
| SVM lineal OVR | 0.733 | 0.042 |
| LogReg OVR | 0.726 | 0.036 |
| SVD + MLP | 0.698 | 0.047 |

La SVM lineal obtiene el mejor rendimiento final en test. La regresion logistica queda cerca y se mantiene como modelo auditado porque sus coeficientes son mas faciles de interpretar. La MLP se usa como comparacion secundaria y no mejora a los modelos lineales.

## 8. Retrieval de precedentes

El modulo de retrieval busca casos parecidos dentro del conjunto de entrenamiento. La relevancia se aproxima por articulos compartidos.

| Metodo | Split | Consultas | nDCG@10 | Recall@5 | Recall@10 |
|---|---|---:|---:|---:|---:|
| TF-IDF coseno | validation | 250 | 0.882 | 0.952 | 0.964 |
| TF-IDF coseno | test | 250 | 0.867 | 0.944 | 0.972 |
| TF-IDF SVD coseno | validation | 250 | 0.898 | 0.960 | 0.964 |
| TF-IDF SVD coseno | test | 250 | 0.876 | 0.940 | 0.956 |

El resultado principal es **Recall@10 = 0.972** para TF-IDF con coseno en test.

La evaluacion es aproximada. Compartir articulo no prueba equivalencia juridica real. Harian falta juristas para validar la utilidad de los precedentes recuperados.

## 9. Explicabilidad, shift y errores

El proyecto revisa coeficientes globales, explicaciones locales con LIME, shift entre particiones y patrones de error.

| Elemento | Resultado |
|---|---:|
| Etiquetas con terminos globales | 10 |
| Explicaciones locales | 5 |
| Peso global medio absoluto | 2.265 |
| Probabilidad local positiva media | 0.816 |

Shift entre particiones

| Split objetivo | JS | L1 | Coseno | Delta macro-F1 | Delta micro-F1 | Delta Hamming |
|---|---:|---:|---:|---:|---:|---:|
| validation | 0.017 | 0.278 | 0.959 | 0.000 | 0.000 | 0.000 |
| test | 0.026 | 0.320 | 0.951 | 0.006 | -0.015 | 0.005 |

Los valores JS, L1 y coseno comparan distribucion frente a `train`. Los deltas de rendimiento comparan frente a `validation`.

Patrones de error en test

| FP | FN | Casos |
|---:|---:|---:|
| 0 | 0 | 491 |
| 1 | 0 | 189 |
| 0 | 1 | 170 |
| 1 | 1 | 65 |
| 0 | 2 | 36 |
| 2 | 0 | 25 |

LIME y los coeficientes ayudan a auditar el modelo. No prueban causalidad juridica. El analisis de shift compara particiones del dataset. No afirma deriva temporal estricta.

## 10. Artefactos principales

Los notebooks guardan resultados bajo `artifacts/`.

| Artefacto | Contenido |
|---|---|
| `artifacts/metrics/cv_gridsearch_results.csv` | Resultados completos de GridSearchCV. |
| `artifacts/metrics/best_params_by_model.json` | Mejores hiperparametros por modelo. |
| `artifacts/metrics/test_model_comparison_cv.csv` | Comparativa final de modelos en test. |
| `artifacts/metrics/paper_classification_table.csv` | Tabla resumida de clasificacion. |
| `artifacts/metrics/paper_retrieval_table.csv` | Tabla resumida de retrieval. |
| `artifacts/metrics/paper_drift_table.csv` | Tabla resumida de shift. |
| `artifacts/metrics/paper_error_pattern_table.csv` | Patrones de error FP y FN. |
| `artifacts/metrics/paper_xai_table.csv` | Resumen de explicabilidad. |
| `artifacts/metrics/xai_lime_summary.csv` | Explicaciones locales LIME. |
| `artifacts/models/notebook_cv5_logreg_pipeline.joblib` | Pipeline final LogReg. |
| `artifacts/models/notebook_cv5_svm_pipeline.joblib` | Pipeline final SVM. |
| `artifacts/models/notebook_cv5_svd_mlp_pipeline.joblib` | Pipeline final SVD + MLP. |

## 11. Limitaciones

1. TF-IDF no capta toda la semantica juridica.
2. LIME y coeficientes muestran senales del modelo, no pruebas juridicas.
3. Retrieval se evalua por articulos compartidos, no por revision experta.
4. Los falsos negativos pueden ocultar articulos relevantes.
5. Los falsos positivos pueden aumentar la carga de revision.
6. La MLP no mejora a los modelos lineales en este protocolo.
7. El sistema necesita supervision humana.

## 12. Alcance

Este trabajo no automatiza la decision judicial. Su objetivo es apoyar la revision inicial de articulos candidatos y precedentes relevantes ante el TEDH.

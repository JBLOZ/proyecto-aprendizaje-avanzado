# Explicacion del proyecto

Este documento resume el proyecto con lenguaje simple para preparar la entrega y la defensa.

El sistema ayuda a revisar casos del Tribunal Europeo de Derechos Humanos. A partir del texto de los hechos, propone articulos candidatos y busca precedentes parecidos.

No predice sentencias. No decide si hay violacion juridica. La decision final siempre debe ser humana.

## 1. Idea principal

El proyecto usa aprendizaje automatico clasico y auditable.

```text
TF-IDF dentro de Pipeline
modelos One-vs-Rest
GridSearchCV con 5 folds
evaluacion final en test
```

La idea no es crear un juez automatico. La idea es priorizar lectura y facilitar auditoria.

## 2. Protocolo correcto

Este es el punto mas importante para la defensa.

1. Se separa `test` desde el principio.
2. Los hiperparametros se seleccionan solo con `train`.
3. La seleccion usa `GridSearchCV` con 5 folds.
4. TF-IDF esta dentro del `Pipeline`.
5. SVD y escalado tambien estan dentro del `Pipeline` en la MLP.
6. El mejor estimador se reentrena con todo `train`.
7. `test` se usa una sola vez para la evaluacion final.

No se ajustan hiperparametros, umbrales ni arquitectura mirando `test`.

`validation` queda como particion oficial e informativa. No decide el modelo final en el protocolo corregido.

## 3. Archivos principales

| Archivo | Funcion |
|---|---|
| `README.md` | Resume el proyecto, ejecucion y resultados. |
| `memoria.tex` | Contiene la memoria final. |
| `schema.sql` | Define la base SQLite. |
| `project_utils.py` | Contiene funciones comunes. |
| `00_ingesta_y_esquema_relacional.ipynb` | Carga datos y crea SQLite. |
| `01_EDA.ipynb` | Analiza datos, splits y desbalanceo. |
| `02_modelado_multilabel.ipynb` | Entrena pipelines con GridSearchCV y evalua test. |
| `03_retrieval_de_precedentes.ipynb` | Busca precedentes similares. |
| `04_xai_drift_y_errores.ipynb` | Revisa explicabilidad, shift y errores. |
| `05_guion_informe_y_defensa.ipynb` | Consolida evidencias para la defensa. |

## 4. Datos

El dataset es **LexGLUE ECtHR Task B**.

| Elemento | Valor |
|---|---:|
| Casos totales | 11000 |
| Train | 9000 |
| Validation | 1000 |
| Test | 1000 |
| Articulos posibles | 10 |
| Pares positivos caso articulo | 15991 |
| Parrafos almacenados | 262580 |

Etiquetas usadas

```text
2, 3, 5, 6, 8, 9, 10, 11, 14, P1-1
```

Es un problema multietiqueta. Un caso puede tener varios articulos positivos.

## 5. Modelos comparados

| Modelo | Papel |
|---|---|
| Baseline de articulo frecuente | Punto de partida simple. |
| Regresion logistica OVR | Modelo auditado. |
| SVM lineal OVR | Mejor rendimiento final en test. |
| TF-IDF + SVD + MLP | Comparacion secundaria no lineal. |

Resultados finales en test

| Modelo | Macro-F1 | Micro-F1 | Hamming loss |
|---|---:|---:|---:|
| SVM lineal OVR | 0.749 | 0.777 | 0.061 |
| LogReg OVR | 0.736 | 0.760 | 0.069 |
| TF-IDF + SVD + MLP | 0.690 | 0.767 | 0.062 |
| Baseline | 0.057 | 0.324 | 0.165 |

Resultados de CV sobre train

| Modelo | Mejor macro-F1 CV |
|---|---:|
| SVM lineal OVR | 0.733 |
| LogReg OVR | 0.726 |
| TF-IDF + SVD + MLP | 0.698 |

Conclusion clara

```text
Mejor rendimiento en test -> SVM lineal
Modelo usado para auditoria -> regresion logistica
MLP -> comparacion secundaria
```

La SVM gana en rendimiento. La regresion logistica se usa para explicar y auditar porque sus coeficientes son directos. La MLP no mejora a los modelos lineales. Probar redes mas grandes mirando test no seria correcto porque romperia el protocolo.

## 6. Retrieval de precedentes

Retrieval significa buscar casos parecidos. El sistema compara el caso de consulta con casos de `train`.

```text
caso de consulta
-> vector TF-IDF
-> similitud coseno
-> ranking de precedentes
```

Resultados en test

| Metodo | nDCG@10 | Recall@5 | Recall@10 |
|---|---:|---:|---:|
| TF-IDF + coseno | 0.867 | 0.944 | 0.972 |
| TF-IDF + SVD + coseno | 0.876 | 0.940 | 0.956 |

El resultado principal es **Recall@10 = 0.972** con TF-IDF + coseno.

La relevancia se aproxima con articulos compartidos. Esto no demuestra que dos casos sean precedentes equivalentes en sentido juridico. Para eso haria falta validacion por juristas.

## 7. Explicabilidad

El proyecto usa dos tipos de explicabilidad.

1. Coeficientes globales de la regresion logistica.
2. Explicaciones locales con LIME.

Resumen de XAI

| Elemento | Valor |
|---|---:|
| Etiquetas con terminos globales | 10 |
| Explicaciones locales | 5 |
| Peso global medio absoluto | 2.265 |
| Probabilidad local positiva media | 0.816 |

Las explicaciones no son razonamiento juridico. Solo muestran que senales textuales usa el modelo.

## 8. Caso practico de auditoria local

Caso verificado en los artefactos.

| Elemento | Valor |
|---|---|
| Caso | `ecthr_task_b_test_000000` |
| Split | `test` |
| Longitud | 4774 tokens |
| Articulo explicado | 10 |
| Score LogReg | 0.96047279757764 |
| Terminos LIME | `journalist`, `broadcasting`, `journalists`, `press` |

El ejemplo tiene sentido porque el articulo 10 se relaciona con libertad de expresion. Aun asi, no prueba que el articulo este juridicamente bien aplicado. Solo muestra trazabilidad del modelo.

## 9. Shift y errores

El proyecto no mide deriva temporal real. La version usada no proporciona un ano oficial de sentencia para todos los casos. Por eso se habla de shift entre particiones.

| Split objetivo | JS | L1 | Coseno | Delta macro-F1 | Delta micro-F1 | Delta Hamming |
|---|---:|---:|---:|---:|---:|---:|
| validation | 0.017 | 0.278 | 0.959 | 0.000 | 0.000 | 0.000 |
| test | 0.026 | 0.320 | 0.951 | 0.006 | -0.015 | 0.005 |

JS, L1 y coseno comparan distribucion frente a `train`. Los deltas de rendimiento comparan frente a `validation`.

Patrones de error en test

| FP | FN | Casos |
|---:|---:|---:|
| 0 | 0 | 491 |
| 1 | 0 | 189 |
| 0 | 1 | 170 |
| 1 | 1 | 65 |
| 0 | 2 | 36 |
| 2 | 0 | 25 |

Un falso positivo es un articulo que el modelo anade de mas. Un falso negativo es un articulo real que el modelo no detecta. Los falsos negativos son delicados porque pueden ocultar articulos relevantes.

## 10. Desbalanceo

No se eliminan casos del articulo 6. Como el problema es multietiqueta, al quitar un caso del articulo 6 tambien se pueden quitar positivos de otros articulos.

El proyecto trata el desbalanceo con tres decisiones.

1. Macro-F1 como metrica principal de seleccion.
2. Pesos de clase cuando procede.
3. Analisis separado de falsos positivos y falsos negativos.

## 11. Frases utiles para defensa

El proyecto no automatiza decisiones judiciales. Sirve para apoyar la revision inicial.

El mejor modelo en test es la SVM lineal One-vs-Rest.

La regresion logistica se usa para auditoria porque permite revisar coeficientes.

El MLP aparece como comparacion secundaria y no mejora a los modelos lineales.

El retrieval ordena casos parecidos. No demuestra que sean precedentes equivalentes en sentido juridico.

Test no se usa para ajustar nada. Solo se usa para la evaluacion final.

## 12. Cierre

La idea final del proyecto es sencilla.

```text
datos juridicos publicos
-> SQLite
-> EDA
-> Pipeline con TF-IDF
-> GridSearchCV con 5 folds en train
-> SVM como mejor modelo final en test
-> LogReg como modelo auditado
-> retrieval de precedentes
-> revision de errores y XAI
```

El sistema es una ayuda para priorizar lectura. No sustituye el criterio juridico.

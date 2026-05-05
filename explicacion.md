He revisado el proyecto. Punto clave: **aquí no se entrenan redes neuronales**. El sistema usa aprendizaje automático clásico y explicable: **TF-IDF + clasificadores lineales One-vs-Rest**. Si en la defensa dices “red neuronal”, te pueden pillar; lo correcto es decir que el modelo funciona como una **capa lineal por artículo**, sin capas ocultas ni embeddings neuronales.

**Qué hace cada archivo**
- [README.md](/Users/jordiblascolozano/Documents/proyecto%20aprendizaje%20avanzado/README.md): explica el objetivo completo: apoyo al análisis inicial de casos TEDH, predicción multietiqueta de artículos, retrieval, XAI, drift y errores.
- [schema.sql](/Users/jordiblascolozano/Documents/proyecto%20aprendizaje%20avanzado/schema.sql): define la base SQLite: `cases`, `case_paragraphs`, `articles`, `case_labels`, `predictions`, `experiment_runs`, `explanations`.
- [00_diagnostico_y_decisiones.ipynb](/Users/jordiblascolozano/Documents/proyecto%20aprendizaje%20avanzado/00_diagnostico_y_decisiones.ipynb): descarga LexGLUE ECtHR Task B, normaliza textos y etiquetas, y crea `data/interim/metadata.db`.
- [01_datos_y_eda.ipynb](/Users/jordiblascolozano/Documents/proyecto%20aprendizaje%20avanzado/01_datos_y_eda.ipynb): analiza tamaño de splits, longitud de casos, frecuencia de artículos, coocurrencias y temporalidad.
- [02_modelado_multilabel.ipynb](/Users/jordiblascolozano/Documents/proyecto%20aprendizaje%20avanzado/02_modelado_multilabel.ipynb): entrena o reutiliza el modelo principal: TF-IDF, regresión logística OVR, SVM OVR y umbrales por etiqueta.
- [03_retrieval_de_precedentes.ipynb](/Users/jordiblascolozano/Documents/proyecto%20aprendizaje%20avanzado/03_retrieval_de_precedentes.ipynb): construye el buscador de precedentes con similitud coseno.
- [04_xai_drift_y_errores.ipynb](/Users/jordiblascolozano/Documents/proyecto%20aprendizaje%20avanzado/04_xai_drift_y_errores.ipynb): carga predicciones, explica pesos, genera LIME, calcula drift y analiza falsos positivos/falsos negativos.
- [05_guion_informe_y_defensa.ipynb](/Users/jordiblascolozano/Documents/proyecto%20aprendizaje%20avanzado/05_guion_informe_y_defensa.ipynb): consolida tablas, figuras y narrativa para informe/defensa.
- [proyecto_main.tex](/Users/jordiblascolozano/Documents/proyecto%20aprendizaje%20avanzado/proyecto_main.tex): memoria final en LaTeX.
- `artifacts/`: contiene modelos `.joblib`, métricas `.csv`, figuras y explicaciones HTML.
- `data/interim/metadata.db`: base de datos central del pipeline.

**Flujo de datos exacto**
1. El dataset entra desde Hugging Face: `lex_glue / ecthr_b`.
2. Cada caso viene como texto o lista de párrafos. El notebook 00 busca campos como `text`, `facts`, `document`, etc.
3. Los párrafos se unen en `text_full`.
4. Se extraen metadatos: `case_id`, `split`, `year`, `n_paragraphs`, `n_tokens`.
5. Las etiquetas del caso se convierten en artículos positivos.
6. Todo se guarda en SQLite:
   - `cases`: 11.000 casos.
   - `case_paragraphs`: 262.580 párrafos.
   - `articles`: 10 artículos.
   - `case_labels`: 15.991 pares positivos caso-artículo.
7. En el notebook 02 se leen `cases` y `case_labels`.
8. Se construye una matriz `Y` de tamaño `11000 x 10`. Cada fila es un caso; cada columna es un artículo. Un `1` significa “este artículo aplica”.
9. Se separan los splits:
   - `train`: 9000 casos.
   - `validation`: 1000 casos.
   - `test`: 1000 casos.
10. El texto `text_full` se transforma con `TfidfVectorizer`:
   - unigramas y bigramas;
   - `min_df=2`;
   - `max_df=0.95`;
   - `sublinear_tf=True`;
   - máximo 60.000 features.
11. El vectorizador solo se ajusta con `train`. Luego transforma `train`, `validation` y `test`.
12. El resultado es una matriz dispersa `X`: filas = casos, columnas = términos/bigramas.
13. Para cada artículo se entrena un clasificador binario independiente. Eso es One-vs-Rest:
   - clasificador 0: artículo 2 sí/no;
   - clasificador 1: artículo 3 sí/no;
   - etc.
14. La regresión logística produce un score por artículo.
15. En `validation` se buscan umbrales individuales por artículo entre `0.10` y `0.90`.
16. Los umbrales finales son:
   `0.25, 0.40, 0.25, 0.50, 0.20, 0.10, 0.20, 0.10, 0.10, 0.30`.
17. Para cada caso:
   - si `score_articulo >= umbral_articulo`, se predice `1`;
   - si no, se predice `0`.
18. Las predicciones se guardan en `predictions` con:
   - `y_true_json`;
   - `y_pred_json`;
   - `scores_json`.

**Cómo “funciona” el modelo**
No hay neuronas. Lo que hay es esto:

```text
texto del caso
→ vector TF-IDF
→ 10 clasificadores lineales
→ 10 scores
→ 10 umbrales
→ vector multietiqueta final
```

Para cada artículo, la regresión logística calcula algo parecido a:

```text
score bruto = peso_1 * termino_1 + peso_2 * termino_2 + ... + bias
probabilidad = sigmoid(score bruto)
```

Si términos como `detention`, `release` o `custody` aparecen con TF-IDF alto, empujan el artículo 5. Si aparecen `hearing`, `proceedings`, `judgment`, empujan el artículo 6. Eso se puede auditar porque los pesos del modelo son visibles.

**Resultados principales**
En clasificación, el modelo principal `tfidf_logreg_threshold_tuned` consigue:
- `test macro-F1`: 0.672
- `test micro-F1`: 0.736
- `test Hamming loss`: 0.0752

En retrieval:
- TF-IDF + coseno en test tiene `Recall@10 = 0.968`.
- SVD + coseno en test tiene `Recall@10 = 0.956`.

**Frase buena para defensa**
“El sistema no usa una red neuronal secuencial. Cada caso se representa como un vector TF-IDF de unigramas y bigramas. Sobre esa matriz entrenamos clasificadores lineales One-vs-Rest, uno por artículo. Cada clasificador devuelve un score, y después aplicamos umbrales ajustados en validación para obtener una salida multietiqueta revisable.”

**Retrieval** significa **recuperación de información**: dado un caso nuevo, el sistema busca en una colección de casos anteriores cuáles son los más parecidos.

En tu proyecto, retrieval es la parte de **recuperación de precedentes jurídicos**.

El flujo es:

```text
caso nuevo
→ se convierte a vector TF-IDF
→ se compara con casos antiguos de train
→ se calcula similitud coseno
→ se devuelven los casos más parecidos
```

Ejemplo simple:

```text
Caso nuevo:
"detención provisional, falta de revisión judicial, duración excesiva"

El sistema busca casos antiguos con vocabulario parecido:
1. caso sobre detention + release
2. caso sobre custody + judicial review
3. caso sobre remand detention
```

No está “prediciendo” artículos ahí. Está haciendo una búsqueda ordenada: **este caso se parece más a estos otros casos**.

En tu proyecto hay dos variantes:

1. **TF-IDF + coseno**
   - Representa cada caso como bolsa de palabras/bigramas ponderados.
   - Compara la similitud entre vectores.
   - Es transparente y fácil de explicar.

2. **TF-IDF + SVD + coseno**
   - Primero reduce la dimensión del vector TF-IDF.
   - Después compara en un espacio más compacto.
   - No es red neuronal; es una proyección lineal.

La evaluación mira si los casos recuperados comparten al menos un artículo real con la consulta. Por eso usas métricas como `Recall@5`, `Recall@10` y `nDCG@10`.

Frase de defensa:

> Retrieval es el módulo de búsqueda de precedentes. Dado un caso de consulta, no predice una sentencia, sino que ordena casos anteriores por similitud textual y permite revisar precedentes que comparten problemáticas jurídicas cercanas.
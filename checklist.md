# CHECKLIST_ML_END_TO_END.md

## Leyenda
- 🟦 **Preprocesamiento (PDF)**
- 🟧 **Preprocesamiento (EXTRA mío)**
- 🟩 **Entrenamiento y evaluación (ciclo de vida)**
- 🟪 **Experimentación y tracking (ciclo de vida)**
- 🟥 **Despliegue / serving (ciclo de vida)**
- 🟫 **Monitorización y mantenimiento (ciclo de vida)**
- ⬛ **Gobernanza y seguridad (ciclo de vida)**

---

# ✅ Checklist completa end-to-end para implementar un modelo

## 🟪 0) Setup del proyecto (estructura, librerías, config)
- [ ] 🟪 Definir objetivo del modelo (qué predice y para qué se usa)
- [ ] 🟪 Definir métricas de éxito (mínimo aceptable + métricas secundarias)
- [ ] 🟪 Definir constraints (latencia, coste, interpretabilidad, privacidad)
- [ ] 🟪 Crear estructura de proyecto (`data/raw`, `data/processed`, `src`, `models`, `results`, `reports`)
- [ ] 🟪 Crear entorno y fijar versiones (`uv/venv/conda`) + lockfile
- [ ] 🟪 Importar librerías necesarias (pandas/numpy/sklearn + extras)
- [ ] 🟪 Config central (YAML/ENV): paths, TARGET, seeds, columnas, features, etc.
- [ ] 🟪 Logging básico (INFO/WARN) y guardado de outputs
- [ ] 🟪 Semillas: `random_state` / numpy seed (reproducibilidad)

---

## 🟪 1) Carga y validación de datos (ingesta)
- [ ] 🟪 Cargar dataset (CSV/Parquet/SQL/API)
- [ ] 🟪 Validar esquema: columnas esperadas, tipos, categorías, rangos básicos
- [ ] 🟪 Normalizar “nulos raros” (`"NA"`, `"?"`, `""`, etc.)
- [ ] 🟪 Separar `df_raw` (intocable) y `df_work` (trabajo)
- [ ] 🟪 Guardar snapshot/version de datos (hash/fecha/partición) si aplica

---

## 🟦🟧 2) Datos y preprocesamiento (data-centric)
### 2.1) EDA (antes de tocar nada)
- [ ] 🟦 Revisar **shape (filas/columnas)**
- [ ] 🟦 Revisar tipos de variables y estructura del dataset
- [ ] 🟦 Revisar target: distribución y posible desbalance
- [ ] 🟦 Detectar missing/outliers y revisar sentido de variables
- [ ] 🟧 Chequeos de coherencia: rangos válidos, unidades, reglas de negocio

### 2.2) Limpieza
- [ ] 🟦 Duplicados, inconsistencias, errores de captura, variables irrelevantes
- [ ] 🟧 Normalización fuerte de categorías (sinónimos, may/min, typos)

### 2.3) Missing values
- [ ] 🟦 Drop filas/columnas, imputación simple (media/mediana/moda), constante
- [ ] 🟦 Imputación por grupos + missing-flag
- [ ] 🟦 KNN Imputer / IterativeImputer
- [ ] 🟦 Series temporales: ffill/bfill/interpolate (con cuidado)
- [ ] 🟧 Auditoría post-imputación (distribuciones antes/después)

### 2.4) Outliers
- [ ] 🟦 Detección (IQR, Z-score) y tratamiento (trimming/clipping/reemplazo robusto)
- [ ] 🟧 Winsorizing por percentiles

### 2.5) Categóricas (encoding)
- [ ] 🟦 Label/Ordinal, OHE, Frequency encoding; Target encoding
- [ ] 🟧 Feature hashing (alta cardinalidad)
- [ ] 🟧 Política de raras + “unknown categories” (producción)

### 2.6) Escalado / transformaciones
- [ ] 🟦 Escalado (Standard/MinMax/Robust)
- [ ] 🟧 QuantileTransformer / PowerTransformer (cuando mejora modelos sensibles)

### 2.7) Feature engineering / selección / reducción
- [ ] 🟦 Creación de variables (interacciones, ratios, fechas, agregaciones)
- [ ] 🟦 Selección **Filter** (correlación, chi², ANOVA, mutual information)
- [ ] 🟦 Selección **Wrapper**:
  - [ ] 🟦 RFE (Recursive Feature Elimination)
  - [ ] 🟦 **RFECV** (RFE con validación cruzada para elegir nº óptimo de features)
  - [ ] 🟦 Forward Selection / Backward Elimination
- [ ] 🟦 Selección **Embedded**:
  - [ ] 🟦 Lasso / Elastic Net (L1/L1+L2)
  - [ ] 🟦 Importancias de árboles (Gini/Permutation como apoyo)
- [ ] 🟦 Explainability como apoyo a selección: **SHAP**
- [ ] 🟦 Reducción dimensional: PCA/LDA, Kernel PCA, t-SNE, UMAP, autoencoders
- [ ] 🟧 Permutation importance (selección robusta y más fiable que importancias internas en algunos casos)

### 2.8) Balanceo de clases
- [ ] 🟦 Oversampling/undersampling, SMOTE/ADASYN/Borderline-SMOTE, Tomek/NearMiss, combinados, BRF/EasyEnsemble/RUSBoost
- [ ] 🟧 Ajuste de threshold por objetivo (max recall / min FP), no solo 0.5

### 2.9) Evaluación ligada al preprocesamiento
- [ ] 🟦 Validación cruzada
- [ ] 🟦 Data leakage (evitar)
- [ ] 🟦 Pipelines (usar)

---

## 🟩 3) Split (train/val/test) y estrategia de evaluación
- [ ] 🟩 Separar `X` e `y`
- [ ] 🟩 Aplicar `train_test_split`
  - [ ] 🟩 `random_state` fijo
  - [ ] 🟩 `stratify=y` si clasificación
- [ ] 🟩 Si aplica: split por grupos (clientes/usuarios) o temporal (series)
- [ ] 🟩 Definir validación: holdout + CV (si procede)
- [ ] 🟩 Definir baseline métrico (modelo simple “tonto pero honesto”)

---

## 🟩 4) Entrenamiento (pipeline + modelos)
- [ ] 🟩 Construir `Pipeline/ColumnTransformer` (preprocesado + modelo)
- [ ] 🟩 Entrenar baseline con pipeline
- [ ] 🟩 Entrenar candidatos (2–4 familias; no un zoo)
- [ ] 🟩 Evaluar con métricas objetivo + matriz de confusión + curvas (ROC/PR si aplica)
- [ ] 🟩 Calibración de probabilidades (si vas a decidir con umbrales)
- [ ] 🟩 Ajuste de umbral (threshold) con validación o CV
- [ ] 🟩 Análisis de errores (segmentos, falsos positivos/negativos, patrones)

---

## 🟩 5) Optimización de hiperparámetros (GridSearch y Optuna)
### 5.1) GridSearchCV / RandomizedSearchCV
- [ ] 🟩 Definir espacio de búsqueda (parámetros + rangos razonables)
- [ ] 🟩 Elegir `scoring` alineado a objetivo
- [ ] 🟩 Ejecutar `GridSearchCV` (pequeño/controlado) o `RandomizedSearchCV` (espacio grande)
- [ ] 🟩 Reentrenar el mejor modelo en train completo (según protocolo)
- [ ] 🟩 Evaluar en test final (una sola vez)

### 5.2) Optuna
- [ ] 🟩 Definir `objective(trial)` (sugerencias + CV + métrica objetivo)
- [ ] 🟩 Definir nº de trials y estrategia (TPE suele bastar)
- [ ] 🟩 Guardar best params + best score + seed
- [ ] 🟩 Reentrenar best model y evaluar en test final

---

## 🟪 6) Tracking, artefactos y reporte
- [ ] 🟪 Registrar runs (params, métricas, tiempo, seed)
- [ ] 🟪 Guardar artefactos:
  - [ ] 🟪 pipeline entrenado (joblib/pkl)
  - [ ] 🟪 schema / lista de columnas / orden
  - [ ] 🟪 plots: CM/ROC/PR, importancias/SHAP, etc.
- [ ] 🟪 Generar reporte final (qué se probó, qué ganó, por qué)

---

## 🟥 7) Despliegue / Serving
- [ ] 🟥 Empaquetar el **pipeline completo** como una unidad (entrada → salida)
- [ ] 🟥 Definir contrato I/O (schema): columnas obligatorias, tipos, defaults
- [ ] 🟥 Robustez en inferencia:
  - [ ] 🟥 unknown categories
  - [ ] 🟥 missing values
  - [ ] 🟥 orden/ausencia de columnas
- [ ] 🟥 Elegir modo: batch (offline) o API (online)
- [ ] 🟥 Smoke tests con datos reales (predice sin romperse)

---

## 🟫 8) Monitorización y mantenimiento
- [ ] 🟫 Monitorizar calidad de datos (missing, rangos, cardinalidad, schema)
- [ ] 🟫 Monitorizar drift (data drift + concept drift si puedes)
- [ ] 🟫 Monitorizar rendimiento (cuando haya ground truth)
- [ ] 🟫 Alertas (rotura de schema, subida de NaN, caída de métrica)
- [ ] 🟫 Plan de retraining (por tiempo, drift o degradación)
- [ ] 🟫 Auditoría de predicciones (muestras, casos límite, explicaciones)

---

## ⬛ 9) Gobernanza, seguridad y compliance
- [ ] ⬛ PII: minimización, enmascarado, retención
- [ ] ⬛ Accesos y secretos (tokens/keys)
- [ ] ⬛ Trazabilidad: dataset+code+modelo → predicción
- [ ] ⬛ Documentación mínima: objetivo, datos, métricas, límites, riesgos

# Seminario de Investigación II — Experimento EC1 + EC2 + EC3

## Evaluación experimental del impacto del sentimiento financiero derivado mediante PLN en la calidad del ranking de un sistema de recomendación de activos cripto

**Autor:** Jhonny Paul Pinto Phillips  
**Programa:** Maestría en Ciencia de Datos e Inteligencia Artificial  
**Universidad:** Universidad Católica Boliviana “San Pablo”  
**Docente:** Mgr. Ismael A. Delgado H.  
**Fecha:** septiembre de 2026

---

## 1. Descripción del proyecto

Este repositorio contiene el experimento reproducible desarrollado para **Seminario de Investigación II**, integrando tres componentes metodológicos:

- **EC1 — Machine Learning / Deep Learning:** estabilidad e interpretabilidad de determinantes predictivos en mercados cripto.
- **EC2 — Sistemas de recomendación:** ranking personalizado de activos cripto mediante señales cuantitativas de mercado.
- **EC3 — Procesamiento del Lenguaje Natural:** extracción de sentimiento financiero mediante modelos clásicos y Transformers.

El objetivo central es evaluar si la incorporación de una señal de sentimiento financiero obtenida mediante PLN mejora la calidad del ranking **Top-K** de un sistema de recomendación de activos cripto.

---

## 2. Pregunta de investigación

> **¿En qué medida la incorporación de una señal de sentimiento financiero obtenida mediante modelos de Procesamiento del Lenguaje Natural influye en la calidad y estabilidad del ranking Top-K de un sistema de recomendación de activos cripto bajo diferentes regímenes de mercado?**

### Variables principales

- **Variable independiente:** incorporación y tipo de señal de sentimiento financiero.
- **Variable dependiente principal:** calidad del ranking Top-K, medida principalmente mediante **NDCG@10**.
- **Variables complementarias:** Recall@10, MAP@10, Jaccard y turnover.
- **Contexto:** régimen de mercado definido mediante momentum histórico de Bitcoin.

---

## 3. Datos utilizados

### 3.1. DLT-Sentiment-News

Dataset principal para el experimento de clasificación de sentimiento financiero.

- Registros originales: **23.301**
- Registros después de eliminar duplicados exactos: **23.293**
- Periodo: **2021-01-01 a 2025-05-23**
- Clases:
  - `0`: neutral
  - `1`: bearish
  - `2`: bullish

Distribución limpia:

| Clase | Observaciones | Porcentaje |
|---|---:|---:|
| Neutral | 9.713 | 41,70 % |
| Bullish | 7.598 | 32,62 % |
| Bearish | 5.982 | 25,68 % |

### 3.2. Financial PhraseBank

Se descargó como benchmark financiero secundario para comparación conceptual y validación metodológica.

- Total: **4.840 frases**
- Train: **3.872**
- Validation: **484**
- Test: **484**

### 3.3. Datos de mercado

Precios diarios obtenidos mediante `yfinance` para 15 activos cripto:

`BTC`, `ETH`, `BNB`, `XRP`, `SOL`, `ADA`, `DOGE`, `AVAX`, `LINK`, `DOT`, `LTC`, `BCH`, `XLM`, `TRX`, `ETC`.

Periodo descargado:

- **2024-11-01 a 2025-06-14**
- Cobertura de mercado para el periodo experimental 2025: **100 %**

---

## 4. Diseño temporal

Para evitar fuga de información se utilizó una partición estrictamente temporal:

| Split | Periodo | N |
|---|---|---:|
| Train | 2021–2023 | 18.619 |
| Validation | 2024 | 3.451 |
| Test | 2025-01-01 a 2025-05-23 | 1.223 |

El conjunto de **test permaneció congelado** y no se utilizó para selección de hiperparámetros.

---

## 5. Modelos de PLN evaluados

Se compararon tres enfoques:

1. **TF-IDF + LinearSVC**
2. **DistilBERT**
3. **FinBERT**

La métrica principal fue **Macro-F1**.

### Resultados en test

| Modelo | Accuracy | Macro-F1 | Tiempo total de inferencia |
|---|---:|---:|---:|
| TF-IDF + LinearSVC | 0,4530 | 0,4444 | 0,125 s |
| DistilBERT | 0,4841 | **0,4754** | 9,796 s |
| FinBERT | 0,4456 | 0,4429 | 20,742 s |

**DistilBERT** fue seleccionado previamente sobre el conjunto de validación y posteriormente obtuvo el mejor desempeño en test.

### Comparaciones estadísticas

Bootstrap pareado de Macro-F1 y prueba exacta de McNemar:

| Comparación | Δ Macro-F1 | IC 95 % | p McNemar |
|---|---:|---:|---:|
| TF-IDF vs DistilBERT | -0,0310 | [-0,0602, -0,0030] | 0,0491 |
| TF-IDF vs FinBERT | +0,0015 | [-0,0281, 0,0305] | 0,6703 |
| DistilBERT vs FinBERT | +0,0324 | [0,0118, 0,0526] | 0,0006 |

En este experimento, DistilBERT superó significativamente a los otros dos modelos en el test congelado.

---

## 6. Señal de sentimiento

A partir de las probabilidades predichas por DistilBERT se definió:

\[
S_t = P(\text{bullish}) - P(\text{bearish})
\]

Características del periodo OOS 2025:

- Documentos: **1.223**
- Media de señal documental: **0,1306**
- Desviación estándar: **0,3340**
- Cobertura diaria con noticias: **141 de 143 días (98,60 %)**
- Media diaria: **0,1226**

La señal fue construida únicamente con predicciones fuera de muestra.

---

## 7. Experimento del sistema de recomendación

### 7.1. Baseline T0

Features cuantitativas:

- momentum a 7 días
- momentum a 30 días
- volatilidad a 14 días

### 7.2. Modelo contextual T1

Añade:

\[
\text{sentiment\_exposure}_{i,t}
=
\beta^{sent}_{i,t} \cdot S_t
\]

donde la beta de sentimiento se estima de forma rolling utilizando únicamente información pasada.

### 7.3. Perfiles simulados

| Perfil | Aversión al riesgo |
|---|---:|
| Conservador | 1,00 |
| Moderado | 0,50 |
| Agresivo | 0,15 |

### 7.4. Periodos

- Calibración: **2025-02-01 a 2025-03-08**
- Embargo: **2025-03-09 a 2025-03-15**
- Evaluación: **2025-03-16 a 2025-05-16**
- Fechas de evaluación: **62**

---

## 8. Resultados del recomendador

### NDCG@10

| Perfil | T0 | T1 | Δ |
|---|---:|---:|---:|
| Agresivo | 0,5853 | 0,5942 | +0,0089 |
| Conservador | 0,6256 | 0,6276 | +0,0019 |
| Moderado | 0,5870 | 0,5929 | +0,0058 |

Promedio global:

- **Δ NDCG@10 = +0,00556**
- Bootstrap 95 %: **[-0,00418, 0,01587]**
- Wilcoxon: **p = 0,4752**

Por tanto, bajo este diseño experimental **no existe evidencia estadística suficiente para sostener una mejora global significativa del ranking al incorporar sentimiento financiero**.

El modelo T1 sí muestra pequeñas mejoras promedio, pero también una ligera reducción de estabilidad del ranking, reflejada en mayor turnover.

---

## 9. Resultados por régimen de mercado

Los regímenes fueron definidos mediante terciles del momentum de BTC estimados únicamente en el periodo de calibración.

Para evitar interpretaciones económicas excesivas, en el análisis final se utilizan las etiquetas:

- **lower-momentum**
- **middle-momentum**
- **upper-momentum**

Las diferencias observadas por régimen se consideran **exploratorias**, ya que algunos grupos contienen muy pocas fechas de evaluación.

---

## 10. Conclusiones principales

1. La arquitectura de PLN sí afecta materialmente la calidad de clasificación de sentimiento.
2. DistilBERT obtuvo el mejor Macro-F1 en el test temporal congelado.
3. Un mejor modelo de PLN no implica automáticamente una mejora estadísticamente significativa del sistema de recomendación.
4. El modelo contextual T1 mejora ligeramente el NDCG@10 promedio, pero la diferencia global no resulta significativa.
5. La incorporación de sentimiento modifica la composición del ranking y reduce ligeramente su estabilidad.
6. La heterogeneidad observada entre regímenes debe interpretarse de forma exploratoria y requiere una muestra temporal mayor.

---

## 11. Estructura del repositorio

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── figures/
├── models/
├── results/
├── src/
│   ├── train_tfidf_svm.py
│   ├── train_distilbert.py
│   ├── train_finbert.py
│   ├── compare_pln_models.py
│   ├── build_sentiment_signal.py
│   ├── download_market_data.py
│   ├── run_recommender_experiment.py
│   ├── robustness_analysis.py
│   └── generate_final_figures_tables.py
├── tables/
├── requirements.txt
└── README.md
```

> Los checkpoints completos de Transformers pueden excluirse del repositorio por su tamaño. Los resultados, predicciones, métricas y metadatos permiten verificar los principales hallazgos sin reentrenar los modelos.

---

## 12. Reproducibilidad

Entorno principal:

- Python **3.11.9**
- PyTorch **2.14.0+cu126**
- Seed: **42**
- GPU utilizada: NVIDIA GTX 1050 Ti 4 GB

### Instalación

```bash
python -m venv .venv
```

En Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## 13. Ejecución del pipeline

El flujo principal del experimento puede reproducirse en el siguiente orden:

```bash
python src/train_tfidf_svm.py
python src/train_distilbert.py
python src/train_finbert.py
python src/compare_pln_models.py
python src/build_sentiment_signal.py
python src/download_market_data.py
python src/run_recommender_experiment.py
python src/robustness_analysis.py
python src/generate_final_figures_tables.py
```

---

## 14. Presentación web

La presentación interactiva del proyecto está disponible en:

**https://paul-pinto.github.io/Seminario-II/**

Repositorio del experimento:

**https://github.com/paul-pinto/Paper-Seminario-II**

---

## 15. Consideraciones éticas y uso de IA

Se utilizó IA generativa como herramienta de apoyo para:

- estructuración y depuración de scripts;
- revisión metodológica;
- organización de análisis, tablas y figuras;
- redacción y edición del documento académico.

La ejecución de los experimentos, verificación de resultados, selección metodológica, interpretación científica y responsabilidad final corresponden al autor.

La IA generativa no se considera coautora ni fuente primaria de evidencia científica.

---

## 16. Autor

**Jhonny Paul Pinto Phillips**  
Maestría en Ciencia de Datos e Inteligencia Artificial  
Universidad Católica Boliviana “San Pablo”

---

## 17. Citación

Si este repositorio es utilizado como referencia:

```text
Pinto Phillips, J. P. (2026).
Evaluación experimental del impacto del sentimiento financiero
derivado mediante PLN en la calidad del ranking de un sistema
de recomendación de activos cripto.
Universidad Católica Boliviana “San Pablo”.
```

---

## Licencia

Este repositorio se publica con fines académicos, educativos y de reproducibilidad científica.

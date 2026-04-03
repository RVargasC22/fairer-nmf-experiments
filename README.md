# Fairer-NMF: Non-negative Matrix Factorization con Restricciones de Equidad

Implementación y evaluación experimental del algoritmo Fairer-NMF propuesto en Kassab et al. (2024), como proyecto final del curso de Machine Learning avanzado.

El paper original está en arXiv: [2411.09847](https://arxiv.org/abs/2411.09847). La idea central es reformular NMF como un problema min-max: en lugar de minimizar el error de reconstrucción global, se minimiza el **peor error de grupo**, lo que da representaciones más equitativas entre subpoblaciones.

---

## Por qué esto importa

NMF estándar minimiza el error agregado. El problema es que cuando los grupos tienen distribuciones distintas (por tamaño, varianza o estructura latente), los grupos minoritarios o más "difíciles" suelen tener errores de reconstrucción sistemáticamente más altos. Esto importa cuando usas los embeddings para algo posterior: clasificación, recomendación, análisis exploratorio. Un embedding de mala calidad para un grupo reproduce y amplifica sesgos.

Fairer-NMF ataca esto directamente sin agregar un hiperparámetro de penalización: la formulación min-max es la restricción en sí.

---

## Estructura del proyecto

```
.
├── fairer_nmf.py          # Implementación de los 3 algoritmos del paper
├── fair_baselines.py      # Baselines: Individual NMF, Fair PCA, Reweighted NMF
├── datasets.py            # Carga y preprocesamiento de los 5 datasets
├── main.py                # Experimento principal (3 algos × 5 datasets)
├── plot_results.py        # 7 gráficas de resultados principales
├── academic_analysis.py   # 12 análisis adicionales (robustez, Pareto, downstream, etc.)
└── results/
    ├── checkpoints/       # Checkpoints PKL por dataset y algoritmo
    ├── plots/             # Todas las figuras generadas (01-20)
    ├── academic/          # PKLs y CSVs de análisis académicos
    └── summary.csv        # Métricas consolidadas
```

---

## Algoritmos implementados

### Del paper

**Algorithm 2 — Alternating Minimization (AM):** En cada iteración resuelve el subproblema de H como un SOCP usando CVXPY/SCS (restricción de la norma de Frobenius por grupo), y actualiza cada fila de W_i con NNLS. Es el método exacto del paper. Lento pero óptimo.

**Algorithm 3 — Multiplicative Updates (MU):** Reformula el problema con una matriz bloque-diagonal ponderada por los errores de grupo actuales, y aplica las reglas de actualización multiplicativa clásicas de NMF. Mucho más rápido, converge en 20-50 iteraciones en la mayoría de datasets.

**Algorithm 1 — Estimación de error base:** Monte Carlo sobre NMF individual por grupo para estimar ε_i de referencia. Se usa internamente para normalizar los errores.

### Baselines adicionales

- **Individual NMF por grupo:** cada grupo tiene su propio diccionario H_i. Upper bound de fairness — máxima flexibilidad, sin representación compartida.
- **Fair PCA (Samadi et al. 2018, aproximación):** subespacio PCA compartido via reweighting iterativo de la covarianza con exponentiated gradient. Permite valores negativos, no es NMF.
- **Reweighted NMF:** oversampling iterativo del grupo con mayor error. Baseline heurístico simple.

---

## Datasets

| Dataset | n | m | Grupos | Atributo sensible |
|---|---|---|---|---|
| Heart Disease (UCI) | 297 | 12 | Female / Male | Sexo |
| German Credit | 1000 | 19 | Female / Male | Sexo |
| Adult Census | 2000 | 43 | Female / Male | Sexo |
| Bank Marketing | 2000 | 15 | Married / Single / Divorced | Estado civil |
| 20 Newsgroups | 1000 | 300 | 6 categorías temáticas | Categoría |

---

## Resultados principales

| Dataset | NMF max err | MU max err | AM max err | MU fairness gain | AM fairness gain |
|---|---|---|---|---|---|
| Heart Disease | 0.073 | 0.071 | 0.071 | +2.7% | +2.7% |
| German Credit | 0.139 | 0.142 | 0.150 | -2.2% | -8.2% |
| Adult Census | 0.385 | 0.363 | 0.385 | +5.7% | +0.1% |
| Bank Marketing | 0.084 | 0.087 | 0.087 | -3.4% | -3.4% |
| 20 Newsgroups | 0.912 | 0.916 | 0.899 | -0.4% | +1.5% |

Los números más interesantes: MU gana claramente en Adult Census (el dataset con mayor disparidad inicial, 0.107). En datasets donde la disparidad base ya es pequeña (German Credit: 0.001, Bank Marketing: 0.005), el algoritmo no tiene mucho margen y a veces introduce ruido. AM destaca en 20 Newsgroups, que tiene 6 grupos — consistente con que SOCP es más preciso cuando hay más restricciones que gestionar.

---

## Cómo reproducir

```bash
# Instalar dependencias
pip install numpy scipy scikit-learn cvxpy matplotlib pandas

# Experimento principal (puede tardar varios minutos por AM en datasets grandes)
python main.py

# Gráficas principales
python plot_results.py

# Análisis académicos completos (multi-seed, Pareto, downstream, t-SNE, etc.)
python academic_analysis.py
```

Los scripts tienen checkpoint automático: si se interrumpen, reanudan desde el último dataset completado. AM guarda checkpoint después de cada iteración individual.

---

## Análisis realizados

Más allá de la comparación básica:

- **Robustez estadística:** 7 semillas distintas + test de Wilcoxon signed-rank para significancia
- **Frontera de Pareto:** barrido de ranks r ∈ {2,3,4,5,6,8,10} para trazar el trade-off fairness/accuracy
- **Fairness downstream:** regresión logística sobre los embeddings W, 5-fold cross-validation, demographic parity gap
- **t-SNE de embeddings:** visualización 2D de W por grupo para NMF, MU y AM
- **Interseccionalidad:** Sexo × Edad en Adult Census → 4 grupos cruzados
- **Sensibilidad al número de grupos K:** particiones aleatorias de K=2 a K=8 en Adult Census
- **Escalabilidad:** tiempo de cómputo vs tamaño del dataset para ambos algoritmos

---

## Observaciones

MU converge en general antes de las 50 iteraciones, así que las 300 del experimento son excesivas para la mayoría de casos (se mantienen por reproducibilidad). AM es entre 10x y 100x más lento según el dataset, lo que lo hace impracticable para datos grandes sin optimizaciones adicionales.

La mejora de fairness no es universal: depende de cuánta disparidad exista de entrada. Cuando NMF estándar ya es equitativo, Fairer-NMF no tiene nada que mejorar y puede introducir pequeñas degradaciones por el cambio de formulación.

---

## Referencia

Kassab, R., Mangold, C., Richard, G., & Rudi, A. (2024). *Towards a Fairer Non-negative Matrix Factorization*. arXiv:2411.09847.

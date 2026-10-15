# ML2 Assignment — Clustering, Classification & Boosting

Three datasets, three task types, multiple algorithms per task, best result
per category tabulated below.

## Setup

```bash
uv venv                  # create .venv (already done in this repo)
uv pip install -r requirements.txt --python .venv/bin/python
```

Notebooks were generated from `src/*.py` (jupytext "percent" format) and
executed headlessly:

```bash
uv run --python .venv/bin/python -m jupytext --to notebook src/<name>.py -o notebooks/<n>.ipynb
uv run --python .venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/<n>.ipynb
```

## Datasets & Tasks

| Dataset | Rows | Task | Target |
|---|---|---|---|
| `Mall_Customers.csv` | 200 | Clustering (unsupervised) | — (segments) |
| `Social_Network_Ads.csv` | 400 | Classification (binary) | `Purchased` |
| `insurance_pre.csv` | 1339 | Boosting / Regression | `charges` |

## Results

| Dataset | Task | Algorithm | Metric | Score | Best |
|---|---|---|---|---|---|
| Mall_Customers.csv | Clustering | KMeans | Silhouette | 0.4208 (k=10 auto) | |
| Mall_Customers.csv | Clustering | KMeans | Silhouette | 0.3041 (k=5 fixed) | |
| Mall_Customers.csv | Clustering | Agglomerative | Silhouette | 0.4176 (k=10 auto) | |
| Mall_Customers.csv | Clustering | Agglomerative | Silhouette | 0.2870 (k=5 fixed) | |
| Mall_Customers.csv | Clustering | **DBSCAN** | Silhouette | **0.4746** (auto, 24.5% noise) | ✅ |
| Mall_Customers.csv | Clustering | DBSCAN | Silhouette | 0.2963 (k=5 fixed) | |
| Mall_Customers.csv | Clustering | GaussianMixture | Silhouette | 0.3972 (k=10 auto) | |
| Mall_Customers.csv | Clustering | GaussianMixture | Silhouette | 0.2216 (k=5 fixed) | |
| Social_Network_Ads.csv | Classification | LogisticRegression | ROC-AUC | 0.9106 | |
| Social_Network_Ads.csv | Classification | KNN | ROC-AUC | 0.9180 | |
| Social_Network_Ads.csv | Classification | **SVM (RBF)** | ROC-AUC | **0.9549** | ✅ |
| Social_Network_Ads.csv | Classification | DecisionTree | ROC-AUC | 0.8759 | |
| Social_Network_Ads.csv | Classification | RandomForest | ROC-AUC | 0.9238 | |
| Social_Network_Ads.csv | Classification | NaiveBayes | ROC-AUC | 0.9475 | |
| insurance_pre.csv | Boosting | **GradientBoosting** | R² | **0.8785** | ✅ |
| insurance_pre.csv | Boosting | AdaBoost | R² | 0.8270 | |
| insurance_pre.csv | Boosting | XGBoost | R² | 0.8439 | |
| insurance_pre.csv | Boosting | LightGBM | R² | 0.8651 | |

Full machine-readable table: [`results/results_summary.csv`](results/results_summary.csv).

### Notes on methodology
- **Clustering**: silhouette score was maximized over k=2..10 for KMeans/
  Agglomerative/GMM. DBSCAN's eps/min_samples were searched too, but capped
  at ≤25% noise points so its score stays comparable to algorithms that
  cluster the whole dataset. A fixed k=5 run is also reported for each
  algorithm since it's the standard interpretable segmentation for this
  dataset (the silhouette-maximizing k was often 10, which is harder to
  interpret as business segments).
- **Classification**: winner picked by ROC-AUC (threshold-independent).
  KNN has the best accuracy/F1 but SVM (RBF) has the best ROC-AUC.
- **Boosting**: winner picked by R² on a held-out 20% test set.

## Project layout

```
src/            jupytext "percent" .py sources (source of truth)
notebooks/      executed .ipynb notebooks (generated from src/)
results/        results_summary.csv + plots (confusion matrix, clusters, pred-vs-actual)
```

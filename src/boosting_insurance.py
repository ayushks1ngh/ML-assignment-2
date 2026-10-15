# %% [markdown]
# # Boosting / Regression — insurance_pre.csv
#
# Goal: predict insurance `charges` (continuous) from age, sex, bmi, children,
# smoker. Compare several boosting algorithms and record the best one.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

RESULTS_PATH = Path("../results/results_summary.csv")
DATASET_NAME = "insurance_pre.csv"
TASK = "Boosting"
RANDOM_STATE = 42

# %% [markdown]
# ## 1. Load & explore

# %%
df = pd.read_csv("../insurance_pre.csv")
print(df.shape)
df.head()

# %%
df.info()
df.isnull().sum()

# %%
df.describe()

# %% [markdown]
# ## 2. Preprocessing
# Encode categoricals (sex, smoker), scale numeric features, split train/test.

# %%
data = df.copy()
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
data["sex"] = le_sex.fit_transform(data["sex"])        # female/male -> 0/1
data["smoker"] = le_smoker.fit_transform(data["smoker"])  # no/yes -> 0/1

feature_cols = ["age", "sex", "bmi", "children", "smoker"]
X = data[feature_cols]
y = data["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ## 3. Train multiple boosting algorithms

# %%
models = {
    "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostRegressor(random_state=RANDOM_STATE),
    "XGBoost": XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
    "LightGBM": LGBMRegressor(random_state=RANDOM_STATE, verbose=-1),
}

runs = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    runs[name] = {"model": model, "r2": r2, "rmse": rmse, "mae": mae, "preds": preds}
    print(f"{name:20s} R2={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")

# %% [markdown]
# ## 4. Pick the best model (by R²) and inspect predicted vs actual charges

# %%
best_name = max(runs, key=lambda n: runs[n]["r2"])
best_run = runs[best_name]
print(f"Best algorithm: {best_name} (R2={best_run['r2']:.4f})")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_run["preds"], alpha=0.6)
lims = [min(y_test.min(), best_run["preds"].min()), max(y_test.max(), best_run["preds"].max())]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual charges")
plt.ylabel("Predicted charges")
plt.title(f"Predicted vs Actual — {best_name} (R2={best_run['r2']:.4f})")
plt.legend()
plt.savefig("../results/boosting_best_pred_vs_actual.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Record results

# %%
rows = [
    {"Dataset": DATASET_NAME, "Task": TASK, "Algorithm": name, "Metric": "R2",
     "Score": round(r["r2"], 4),
     "Params": f"rmse={r['rmse']:.2f}, mae={r['mae']:.2f}"}
    for name, r in runs.items()
]
results_df = pd.DataFrame(rows)
results_df["Best"] = results_df["Algorithm"] == best_name
print(results_df.to_string(index=False))

# %%
RESULTS_PATH.parent.mkdir(exist_ok=True)
if RESULTS_PATH.exists():
    existing = pd.read_csv(RESULTS_PATH)
    existing = existing[existing["Dataset"] != DATASET_NAME]
    combined = pd.concat([existing, results_df], ignore_index=True)
else:
    combined = results_df
combined.to_csv(RESULTS_PATH, index=False)
print(f"Saved to {RESULTS_PATH.resolve()}")

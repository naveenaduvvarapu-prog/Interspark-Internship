"""
╔══════════════════════════════════════════════════════════════════════════╗
║        HOUSE PRICE PREDICTION — Full ML Pipeline Notebook               ║
║  Dataset: kaggle.com/datasets/bhanupratapbiswas/house-price-prediction   ║
║  Author  : Claude (Anthropic)   Date: 2026-06-12                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Notebook is structured as # %% cells — open in VS Code / Jupyter or   ║
║  run end-to-end with:  python house_price_notebook.py                   ║
╚══════════════════════════════════════════════════════════════════════════╝

OUTPUTS produced
  01_eda.png             — EDA charts
  02_evaluation.png      — Model comparison & residual analysis
  house_price_model.joblib — Saved best-model pipeline
  feature_info.json      — Feature schema reference
"""

# %% ── CELL 0 : IMPORTS ──────────────────────────────────────────────────
import warnings, os, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
from scipy import stats as sp_stats

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

OUT  = "."           # Change to your output folder
SEED = 42
np.random.seed(SEED)

# %% ── CELL 1 : LOAD DATA ────────────────────────────────────────────────
# ┌─────────────────────────────────────────────────────────┐
# │  Replace the block below with:                          │
# │    df = pd.read_csv("House Price Prediction.csv")       │
# │  if you have downloaded the Kaggle dataset.             │
# └─────────────────────────────────────────────────────────┘
n = 1500
neighborhoods = ["Downtown", "Suburbs", "Rural", "Waterfront", "Historic"]
house_styles  = ["Ranch", "Colonial", "Victorian", "Cape Cod", "Contemporary"]
conditions    = ["Excellent", "Good", "Average", "Fair", "Poor"]
condition_map = {"Excellent": 5, "Good": 4, "Average": 3, "Fair": 2, "Poor": 1}

df = pd.DataFrame({
    "Id"           : range(1, n + 1),
    "LotArea"      : np.random.lognormal(9.2, 0.5, n).astype(int),
    "YearBuilt"    : np.random.randint(1900, 2023, n),
    "OverallQual"  : np.random.choice(range(1, 11), n,
                         p=[.01,.02,.04,.07,.12,.18,.22,.18,.10,.06]),
    "GrLivArea"    : np.random.lognormal(7.4, 0.35, n).astype(int),
    "TotalBsmtSF"  : np.random.lognormal(6.8, 0.55, n).astype(int),
    "GarageArea"   : np.random.randint(200, 900, n),
    "FullBath"     : np.random.choice([1,2,3,4], n, p=[.20,.55,.20,.05]),
    "BedroomAbvGr" : np.random.choice([1,2,3,4,5], n, p=[.05,.20,.50,.20,.05]),
    "Neighborhood" : np.random.choice(neighborhoods, n),
    "HouseStyle"   : np.random.choice(house_styles, n),
    "OverallCond"  : np.random.choice(conditions, n, p=[.10,.35,.35,.15,.05]),
    "CentralAir"   : np.random.choice(["Y","N"], n, p=[.85,.15]),
})
# patch GarageArea: 10 % of houses have no garage
df["GarageArea"] = np.where(np.random.rand(n) < .10, 0, df["GarageArea"])

# synthetic price signal
df["_cond_num"]   = df["OverallCond"].map(condition_map)
df["_neigh_mult"] = df["Neighborhood"].map(
    {"Waterfront":.35,"Downtown":.15,"Historic":.05,"Suburbs":0,"Rural":-.15})
base = (  0.55*np.log(df["GrLivArea"])
        + 0.25*np.log(df["LotArea"])
        + 0.40*df["OverallQual"]
        + 0.12*df["_cond_num"]
        + 0.008*(df["YearBuilt"]-1900)
        + 0.0003*df["GarageArea"]
        + 0.0002*df["TotalBsmtSF"]
        + 0.06*df["FullBath"]
        + df["_neigh_mult"]
        + (df["CentralAir"]=="Y").astype(int)*0.08)
df["SalePrice"] = np.exp(base + np.random.normal(0,.12,n) + 5.5).astype(int)
df.drop(columns=["_cond_num","_neigh_mult"], inplace=True)

# inject realistic missingness
for col, rate in [("GarageArea",.05),("TotalBsmtSF",.03),
                  ("LotArea",.02),("FullBath",.01)]:
    df.loc[np.random.rand(n) < rate, col] = np.nan

print(f"Shape: {df.shape}")
print(df.head(3))
print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum()>0])
print("\nSalePrice:\n", df["SalePrice"].describe().round(0))

# %% ── CELL 2 : EDA ──────────────────────────────────────────────────────
num_cols_eda = ["SalePrice","GrLivArea","LotArea","OverallQual",
                "YearBuilt","TotalBsmtSF","GarageArea","FullBath"]

fig = plt.figure(figsize=(18, 14))
fig.suptitle("House Price EDA", fontsize=16, fontweight="bold")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=.45, wspace=.35)

ax = fig.add_subplot(gs[0,0])
sns.histplot(df["SalePrice"], bins=40, kde=True, ax=ax, color="#4C72B0")
ax.set_title("SalePrice Distribution"); ax.set_xlabel("SalePrice ($)")

ax = fig.add_subplot(gs[0,1])
sns.histplot(np.log(df["SalePrice"]), bins=40, kde=True, ax=ax, color="#55A868")
ax.set_title("log(SalePrice) — near-normal"); ax.set_xlabel("log(SalePrice)")

ax = fig.add_subplot(gs[0,2])
corr = df[num_cols_eda].corr()
sns.heatmap(corr, mask=np.triu(np.ones_like(corr,bool)),
            annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
            cbar_kws={"shrink":.6}, annot_kws={"size":7})
ax.set_title("Correlation Matrix")
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.tick_params(axis='y', rotation=0,  labelsize=7)

ax = fig.add_subplot(gs[1,0])
ax.scatter(df["GrLivArea"], df["SalePrice"], alpha=.25, s=10, color="#4C72B0")
ax.set_title("GrLivArea vs SalePrice"); ax.set_xlabel("sqft"); ax.set_ylabel("$")

ax = fig.add_subplot(gs[1,1])
sns.boxplot(x="OverallQual", y="SalePrice", data=df, ax=ax,
            palette="viridis", linewidth=0.8)
ax.set_title("OverallQual vs SalePrice")

ax = fig.add_subplot(gs[1,2])
nm = df.groupby("Neighborhood")["SalePrice"].median().sort_values()
ax.barh(nm.index, nm.values/1000, color="#4C72B0")
ax.set_title("Median Price by Neighborhood"); ax.set_xlabel("$K")

ax = fig.add_subplot(gs[2,0])
ax.scatter(df["YearBuilt"], df["SalePrice"], alpha=.2, s=8, color="#C44E52")
ax.set_title("YearBuilt vs SalePrice"); ax.set_xlabel("Year Built")

ax = fig.add_subplot(gs[2,1])
sns.boxplot(x="OverallCond", y="SalePrice", data=df,
            order=["Excellent","Good","Average","Fair","Poor"],
            ax=ax, palette="Set2", linewidth=0.8)
ax.set_title("Condition vs SalePrice"); ax.tick_params(axis='x', rotation=20)

ax = fig.add_subplot(gs[2,2])
miss = df.isnull().sum()[df.isnull().sum()>0].sort_values()
ax.barh(miss.index, miss.values, color="#DD8452")
ax.set_title("Missing Values per Column"); ax.set_xlabel("Count")

plt.savefig(f"{OUT}/01_eda.png", dpi=130, bbox_inches="tight")
plt.close()
print("Saved 01_eda.png")

# %% ── CELL 3 : FEATURE ENGINEERING ──────────────────────────────────────
df_fe = df.copy()

# ── derived features ──────────────────────────────────────────────────────
df_fe["HouseAge"]         = 2024 - df_fe["YearBuilt"]
df_fe["TotalSF"]          = df_fe["GrLivArea"] + df_fe["TotalBsmtSF"].fillna(0)
df_fe["BathPerBed"]       = (df_fe["FullBath"] / df_fe["BedroomAbvGr"]).clip(0, 5)
df_fe["HasGarage"]        = (df_fe["GarageArea"].fillna(0) > 0).astype(int)
df_fe["QualCondInteract"] = df_fe["OverallQual"] * \
                             df_fe["OverallCond"].map(condition_map)
df_fe["LogLotArea"]       = np.log1p(df_fe["LotArea"])
df_fe["LogGrLivArea"]     = np.log1p(df_fe["GrLivArea"])

# ── target: log-transform to reduce skew ─────────────────────────────────
df_fe["LogSalePrice"] = np.log(df_fe["SalePrice"])

DROP  = ["Id","SalePrice","LogSalePrice","YearBuilt","LotArea","GrLivArea"]
feature_cols = [c for c in df_fe.columns if c not in DROP]

X = df_fe[feature_cols]
y = df_fe["LogSalePrice"]

num_feats = X.select_dtypes(include=np.number).columns.tolist()
cat_feats  = X.select_dtypes(include="object").columns.tolist()

print("Numeric features :", num_feats)
print("Categorical features:", cat_feats)

# %% ── CELL 4 : PREPROCESSING PIPELINE ───────────────────────────────────
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_feats),
    ("cat", cat_pipeline, cat_feats),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.20, random_state=SEED)
print(f"Train {X_train.shape[0]} | Test {X_test.shape[0]}")

# %% ── CELL 5 : MODEL TRAINING ────────────────────────────────────────────
models = {
    "Linear Regression" : LinearRegression(),
    "Ridge Regression"  : Ridge(alpha=10.0),
    "Random Forest"     : RandomForestRegressor(
                              n_estimators=300, min_samples_leaf=2,
                              n_jobs=-1, random_state=SEED),
    "Gradient Boosting" : GradientBoostingRegressor(
                              n_estimators=400, learning_rate=.05,
                              max_depth=4, subsample=.8,
                              min_samples_leaf=5, random_state=SEED),
}

results   = {}
pipelines = {}

for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    y_pred = np.exp(pipe.predict(X_test))
    y_true = np.exp(y_test)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    cv   = np.sqrt(-cross_val_score(pipe, X_train, y_train,
                    cv=5, scoring="neg_mean_squared_error")).mean()

    results[name]   = {"RMSE": rmse, "MAE": mae, "R²": r2, "CV_logRMSE": cv}
    pipelines[name] = pipe
    print(f"[{name}]  RMSE=${rmse:>10,.0f}  MAE=${mae:>9,.0f}  "
          f"R²={r2:.4f}  CV={cv:.4f}")

best_name = min(results, key=lambda k: results[k]["RMSE"])
best_pipe = pipelines[best_name]
print(f"\n★ Best: {best_name}")

# %% ── CELL 6 : EVALUATION & RESIDUALS ───────────────────────────────────
y_pred_best = np.exp(best_pipe.predict(X_test))
y_true_arr  = np.exp(y_test.values)
residuals   = y_true_arr - y_pred_best
pct_errors  = residuals / y_true_arr * 100

fig2, axes = plt.subplots(3, 3, figsize=(18,15))
fig2.suptitle("Model Evaluation & Residual Analysis", fontsize=15, fontweight="bold")
colors = ["#4C72B0","#55A868","#C44E52","#8172B3"]
mnames = list(results.keys())

# RMSE bar
ax = axes[0,0]
bars = ax.bar(mnames, [results[m]["RMSE"]/1000 for m in mnames], color=colors)
ax.set_title("RMSE ($K)"); ax.tick_params(axis='x', rotation=15, labelsize=8)
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+.3,
            f"${b.get_height():.1f}K", ha="center", fontsize=8)

# MAE bar
ax = axes[0,1]
ax.bar(mnames, [results[m]["MAE"]/1000 for m in mnames], color=colors)
ax.set_title("MAE ($K)"); ax.tick_params(axis='x', rotation=15, labelsize=8)

# R² bar
ax = axes[0,2]
ax.bar(mnames, [results[m]["R²"] for m in mnames], color=colors)
ax.set_title("R² Score"); ax.set_ylim(0,1)
ax.axhline(.90, color="red", linestyle="--", linewidth=.8)
ax.tick_params(axis='x', rotation=15, labelsize=8)

# Actual vs Predicted
ax = axes[1,0]
lo = min(y_true_arr.min(), y_pred_best.min())/1000
hi = max(y_true_arr.max(), y_pred_best.max())/1000
ax.scatter(y_true_arr/1000, y_pred_best/1000, alpha=.3, s=12, color="#4C72B0")
ax.plot([lo,hi],[lo,hi],"r--",linewidth=1.5)
ax.set_title(f"Actual vs Predicted — {best_name}")
ax.set_xlabel("Actual ($K)"); ax.set_ylabel("Predicted ($K)")

# Residuals vs Predicted
ax = axes[1,1]
ax.scatter(y_pred_best/1000, residuals/1000, alpha=.3, s=12, color="#55A868")
ax.axhline(0, color="red", linestyle="--", linewidth=1.2)
ax.set_title("Residuals vs Predicted"); ax.set_xlabel("Predicted ($K)")

# Residual histogram
ax = axes[1,2]
sns.histplot(residuals/1000, bins=40, kde=True, ax=ax, color="#C44E52")
ax.axvline(0, color="black", linewidth=1)
ax.set_title("Residual Distribution"); ax.set_xlabel("Residual ($K)")

# % Error histogram
ax = axes[2,0]
sns.histplot(pct_errors, bins=40, kde=True, ax=ax, color="#8172B3")
ax.set_title("% Prediction Error"); ax.set_xlabel("% Error")
within_10 = (np.abs(pct_errors) < 10).mean()*100
ax.text(.98,.95, f"Within ±10%: {within_10:.1f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

# Feature importance (RF)
ax = axes[2,1]
rf = pipelines["Random Forest"]
prep = rf.named_steps["prep"]
fnames = (num_feats +
          list(prep.named_transformers_["cat"]
                   .named_steps["encoder"]
                   .get_feature_names_out(cat_feats)))
imp = pd.Series(rf.named_steps["model"].feature_importances_,
                index=fnames).sort_values(ascending=False)[:15]
imp[::-1].plot(kind="barh", ax=ax, color="#4C72B0")
ax.set_title("RF Feature Importances (top 15)"); ax.tick_params(labelsize=7)

# Q-Q plot
ax = axes[2,2]
(osm, osr), (slope, intercept, _) = sp_stats.probplot(residuals/1000)
ax.plot(osm, osr, "o", alpha=.3, ms=3, color="#4C72B0")
ax.plot(osm, slope*np.array(osm)+intercept, "r--", linewidth=1.5)
ax.set_title("Q-Q Plot of Residuals")
ax.set_xlabel("Theoretical Quantiles"); ax.set_ylabel("Sample Quantiles ($K)")

plt.tight_layout()
plt.savefig(f"{OUT}/02_evaluation.png", dpi=130, bbox_inches="tight")
plt.close()
print("Saved 02_evaluation.png")

# Summary table
print("\n── Model Comparison ─────────────────────────────────────────────")
for m, v in results.items():
    print(f"  {m:<22}  RMSE=${v['RMSE']:>10,.0f}  MAE=${v['MAE']:>9,.0f}"
          f"  R²={v['R²']:.4f}  CV={v['CV_logRMSE']:.4f}")
print(f"\n  ★ Best → {best_name}  |  Within ±10%: {within_10:.1f}%")

# %% ── CELL 7 : SAVE MODEL & FEATURE SCHEMA ───────────────────────────────
model_bundle = {
    "pipeline"    : best_pipe,
    "feature_cols": feature_cols,
    "num_cols"    : num_feats,
    "cat_cols"    : cat_feats,
    "model_name"  : best_name,
}
joblib.dump(model_bundle, f"{OUT}/house_price_model.joblib")
print(f"Model saved → house_price_model.joblib")

with open(f"{OUT}/feature_info.json","w") as f:
    json.dump({
        "features"   : feature_cols,
        "engineered" : ["HouseAge","TotalSF","BathPerBed","HasGarage",
                        "QualCondInteract","LogLotArea","LogGrLivArea"],
        "target"     : "log(SalePrice) — use np.exp() to convert back",
        "best_model" : best_name,
    }, f, indent=2)
print("Feature schema saved → feature_info.json")


# ══════════════════════════════════════════════════════════════════════════
# %% ── CELL 8 : PREDICTION CELL (copy-paste to use saved model) ──────────
# ══════════════════════════════════════════════════════════════════════════
"""
STANDALONE PREDICTION USAGE — run this block independently:

  import joblib, numpy as np, pandas as pd

  bundle = joblib.load("house_price_model.joblib")
  pipe   = bundle["pipeline"]

  # Fill in all features after engineering (see list below)
  new_house = pd.DataFrame([{
      "OverallQual"     : 7,
      "TotalBsmtSF"     : 900,
      "GarageArea"      : 440,
      "FullBath"        : 2,
      "BedroomAbvGr"    : 3,
      "HouseAge"        : 25,           # 2024 - YearBuilt
      "TotalSF"         : 2200,         # GrLivArea + TotalBsmtSF
      "BathPerBed"      : 2/3,          # FullBath / BedroomAbvGr
      "HasGarage"       : 1,            # 1 if GarageArea > 0
      "QualCondInteract": 7 * 4,        # OverallQual * OverallCond_num
      "LogLotArea"      : np.log1p(9000),
      "LogGrLivArea"    : np.log1p(1300),
      "Neighborhood"    : "Suburbs",
      "HouseStyle"      : "Colonial",
      "OverallCond"     : "Good",
      "CentralAir"      : "Y",
  }])

  log_price    = pipe.predict(new_house)[0]
  dollar_price = np.exp(log_price)
  print(f"Predicted price: ${dollar_price:,.0f}")
"""

import joblib

bundle    = joblib.load(f"{OUT}/house_price_model.joblib")
pipe      = bundle["pipeline"]

new_house = pd.DataFrame([{
    "OverallQual"     : 7,
    "TotalBsmtSF"     : 900.0,
    "GarageArea"      : 440.0,
    "FullBath"        : 2.0,
    "BedroomAbvGr"    : 3,
    "HouseAge"        : 25,
    "TotalSF"         : 2200,
    "BathPerBed"      : 2/3,
    "HasGarage"       : 1,
    "QualCondInteract": 7 * 4,
    "LogLotArea"      : np.log1p(9000),
    "LogGrLivArea"    : np.log1p(1300),
    "Neighborhood"    : "Suburbs",
    "HouseStyle"      : "Colonial",
    "OverallCond"     : "Good",
    "CentralAir"      : "Y",
}])

predicted_price = np.exp(pipe.predict(new_house)[0])
print(f"\n{'='*45}")
print(f"  ★  Predicted SalePrice: ${predicted_price:>12,.0f}")
print(f"{'='*45}\n")

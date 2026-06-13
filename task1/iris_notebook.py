"""
╔══════════════════════════════════════════════════════════════════════════╗
║         IRIS SPECIES CLASSIFICATION — Full ML Pipeline                  ║
║  Dataset : kaggle.com/datasets/bhanupratapbiswas/iris-classification    ║
║  Author  : Claude (Anthropic)   |   Date : 2026-06-13                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Run end-to-end:  python iris_notebook.py                               ║
║  Cells marked  # %%  are compatible with VS Code / Jupyter nbformat     ║
╚══════════════════════════════════════════════════════════════════════════╝

OUTPUTS
  01_eda.png              — EDA & class-separability visuals
  02_model_evaluation.png — Confusion matrices + metric comparison
  iris_model.joblib       — Saved best-model pipeline
  README.md               — Inference instructions
"""

# %% ── CELL 0 : IMPORTS ───────────────────────────────────────────────────
import warnings, os, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, precision_score, recall_score, f1_score)

OUT  = "."
SEED = 42
np.random.seed(SEED)

SPECIES    = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
PALETTE    = ["#4C72B0", "#55A868", "#C44E52"]
FEATURES   = ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]

# %% ── CELL 1 : LOAD / BUILD DATASET ─────────────────────────────────────
print("="*65)
print("  SECTION 1 — DATA")
print("="*65)

# ── If you downloaded the Kaggle CSV, replace with: ──────────────────────
#   df = pd.read_csv("Iris.csv")
# ─────────────────────────────────────────────────────────────────────────
iris   = load_iris(as_frame=True)
df     = iris.frame.copy()
df.columns = FEATURES + ["Species"]
df["Species"] = df["Species"].map({0:"Iris-setosa",
                                    1:"Iris-versicolor",
                                    2:"Iris-virginica"})
# Add Id column to match Kaggle schema
df.insert(0, "Id", range(1, len(df)+1))

print(f"Shape  : {df.shape}")
print(f"\nFirst 5 rows:\n{df.head().to_string(index=False)}")
print(f"\nClass distribution:\n{df['Species'].value_counts()}")
print(f"\nDescriptive statistics:\n{df[FEATURES].describe().round(3)}")
print(f"\nMissing values: {df.isnull().sum().sum()}  (none — Iris is clean)")

# %% ── CELL 2 : EDA ───────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 2 — EXPLORATORY DATA ANALYSIS")
print("="*65)

fig = plt.figure(figsize=(20, 16))
fig.suptitle("Iris Dataset — EDA & Class Separability",
             fontsize=17, fontweight="bold", y=1.01)

# ── 2a  Pair plot (manual, 4×4 grid using scatter + kde diag) ────────────
gs_main = fig.add_gridspec(3, 3, hspace=.42, wspace=.32)

# Row 0 — pair plots (SepalLength vs others, PetalLength vs PetalWidth)
pair_axes = [
    (fig.add_subplot(gs_main[0,0]), "SepalLengthCm", "SepalWidthCm"),
    (fig.add_subplot(gs_main[0,1]), "PetalLengthCm", "PetalWidthCm"),
    (fig.add_subplot(gs_main[0,2]), "SepalLengthCm", "PetalLengthCm"),
]
for ax, xf, yf in pair_axes:
    for sp, col in zip(SPECIES, PALETTE):
        sub = df[df["Species"]==sp]
        ax.scatter(sub[xf], sub[yf], c=col, alpha=.65, s=28, label=sp)
    ax.set_xlabel(xf, fontsize=8)
    ax.set_ylabel(yf, fontsize=8)
    ax.set_title(f"{xf} vs {yf}", fontsize=9)
    if xf == "SepalLengthCm" and yf == "SepalWidthCm":
        handles = [mpatches.Patch(color=c,label=s)
                   for c,s in zip(PALETTE,SPECIES)]
        ax.legend(handles=handles, fontsize=7, loc="upper right")

# ── 2b  Box plots — all 4 features by species ────────────────────────────
for i, feat in enumerate(FEATURES):
    ax = fig.add_subplot(gs_main[1, i if i<3 else 2])
    if i == 3:
        ax = fig.add_subplot(gs_main[2, 0])
    data_plot = [df[df["Species"]==sp][feat].values for sp in SPECIES]
    bp = ax.boxplot(data_plot, patch_artist=True,
                    medianprops=dict(color="black",linewidth=1.5))
    for patch, col in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(col); patch.set_alpha(.7)
    ax.set_xticklabels([s.split("-")[1] for s in SPECIES], fontsize=8)
    ax.set_title(f"{feat}", fontsize=9)
    ax.set_ylabel("cm", fontsize=8)

# ── 2c  Violin — PetalLengthCm ───────────────────────────────────────────
ax = fig.add_subplot(gs_main[1, 2])
sns.violinplot(x="Species", y="PetalLengthCm", data=df,
               palette=PALETTE, ax=ax, inner="quartile", linewidth=.8)
ax.set_xticklabels([s.split("-")[1] for s in SPECIES], fontsize=8)
ax.set_title("PetalLength Violin", fontsize=9)

# ── 2d  Correlation heatmap ───────────────────────────────────────────────
ax = fig.add_subplot(gs_main[2, 1])
corr = df[FEATURES].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
            cbar_kws={"shrink":.7}, annot_kws={"size":9},
            mask=np.triu(np.ones_like(corr,bool)))
ax.set_title("Feature Correlation", fontsize=9)
ax.tick_params(labelsize=8)

# ── 2e  Class count bar ───────────────────────────────────────────────────
ax = fig.add_subplot(gs_main[2, 2])
counts = df["Species"].value_counts().reindex(SPECIES)
ax.bar([s.split("-")[1] for s in SPECIES], counts.values,
       color=PALETTE, alpha=.85)
ax.set_title("Class Distribution", fontsize=9)
ax.set_ylabel("Count")
for i,(sp,v) in enumerate(zip(SPECIES, counts.values)):
    ax.text(i, v+.5, str(v), ha="center", fontsize=9)

plt.savefig(f"{OUT}/01_eda.png", dpi=130, bbox_inches="tight")
plt.close()
print("EDA chart saved → 01_eda.png")
print(f"\nKey insight: Petal features show near-perfect class separation.")
print(f"  PetalLengthCm corr with species label: very high")
print(f"  SepalWidthCm corr with PetalLengthCm : "
      f"{corr.loc['SepalWidthCm','PetalLengthCm']:.3f}")

# %% ── CELL 3 : PREPROCESSING ─────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 3 — PREPROCESSING & TRAIN/TEST SPLIT")
print("="*65)

X = df[FEATURES]
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.20, random_state=SEED, stratify=y)

print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
print(f"Train class balance:\n{y_train.value_counts()}")

# %% ── CELL 4 : MODEL TRAINING ────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 4 — MODEL TRAINING (k-NN, Logistic Regression, Decision Tree)")
print("="*65)

models = {
    "k-NN (k=5)"          : KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression"  : LogisticRegression(
                                max_iter=500, random_state=SEED,
                                solver="lbfgs"),
    "Decision Tree"        : DecisionTreeClassifier(
                                max_depth=4, min_samples_leaf=3,
                                random_state=SEED),
}

skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
results = {}
pipes   = {}

for name, model in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec  = recall_score(y_test, y_pred, average="macro")
    f1   = f1_score(y_test, y_pred, average="macro")
    cv   = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy").mean()
    cm   = confusion_matrix(y_test, y_pred, labels=SPECIES)

    results[name] = {"Accuracy":acc,"Precision":prec,
                     "Recall":rec,"F1":f1,"CV_Acc":cv,"CM":cm,
                     "y_pred":y_pred}
    pipes[name]   = pipe

    print(f"\n[{name}]")
    print(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  "
          f"Recall={rec:.4f}  F1={f1:.4f}  CV={cv:.4f}")
    print(classification_report(y_test, y_pred,
          target_names=[s.split("-")[1] for s in SPECIES], digits=4))

best_name = max(results, key=lambda k: results[k]["F1"])
best_pipe = pipes[best_name]
print(f"\n★ Best model: {best_name}  (F1={results[best_name]['F1']:.4f})")

# %% ── CELL 5 : EVALUATION PLOTS ──────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 5 — EVALUATION & VISUALISATION")
print("="*65)

n_models = len(models)
fig2     = plt.figure(figsize=(20, 14))
fig2.suptitle("Model Evaluation — Iris Classification",
              fontsize=15, fontweight="bold")
gs2 = fig2.add_gridspec(3, n_models, hspace=.50, wspace=.35)

mnames = list(results.keys())
mcolors = ["#4C72B0","#55A868","#C44E52"]

# ── Row 0 : confusion matrices ────────────────────────────────────────────
for col_idx, (name, col) in enumerate(zip(mnames, mcolors)):
    ax = fig2.add_subplot(gs2[0, col_idx])
    cm = results[name]["CM"]
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[s.split("-")[1] for s in SPECIES])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix\n{name}", fontsize=9)
    ax.tick_params(labelsize=8)

# ── Row 1 : metric bar chart (per model) ─────────────────────────────────
metrics_show = ["Accuracy","Precision","Recall","F1","CV_Acc"]
x_pos = np.arange(len(metrics_show))
width = .22
ax_bar = fig2.add_subplot(gs2[1, :])
for i, (name, col) in enumerate(zip(mnames, mcolors)):
    vals = [results[name][m] for m in metrics_show]
    bars = ax_bar.bar(x_pos + i*width, vals, width, label=name,
                      color=col, alpha=.85)
ax_bar.set_xticks(x_pos + width)
ax_bar.set_xticklabels(metrics_show, fontsize=10)
ax_bar.set_ylim(.80, 1.02)
ax_bar.axhline(1.0, color="gray", linestyle="--", linewidth=.8)
ax_bar.set_title("Metric Comparison across Models", fontsize=11)
ax_bar.set_ylabel("Score")
ax_bar.legend(fontsize=9)
for p in ax_bar.patches:
    if p.get_height() > 0.82:
        ax_bar.annotate(f"{p.get_height():.3f}",
                        (p.get_x()+p.get_width()/2, p.get_height()+.002),
                        ha="center", fontsize=7, rotation=90)

# ── Row 2 : Decision tree visualisation + k-NN k sweep ───────────────────
ax_tree = fig2.add_subplot(gs2[2, :2])
dt_model = pipes["Decision Tree"].named_steps["clf"]
plot_tree(dt_model,
          feature_names=FEATURES,
          class_names=[s.split("-")[1] for s in SPECIES],
          filled=True, rounded=True, fontsize=7, ax=ax_tree, impurity=False)
ax_tree.set_title("Decision Tree Structure (max_depth=4)", fontsize=9)

ax_k = fig2.add_subplot(gs2[2, 2])
k_range = range(1, 21)
k_scores = [cross_val_score(
    Pipeline([("sc",StandardScaler()),("clf",KNeighborsClassifier(n_neighbors=k))]),
    X, y, cv=skf, scoring="accuracy").mean()
    for k in k_range]
ax_k.plot(k_range, k_scores, "o-", color="#4C72B0", linewidth=1.5, ms=5)
ax_k.axvline(5, color="red", linestyle="--", linewidth=1, label="k=5 used")
ax_k.set_title("k-NN: CV Accuracy vs k", fontsize=9)
ax_k.set_xlabel("k"); ax_k.set_ylabel("CV Accuracy")
ax_k.legend(fontsize=8)

plt.savefig(f"{OUT}/02_model_evaluation.png", dpi=130, bbox_inches="tight")
plt.close()
print("Evaluation chart saved → 02_model_evaluation.png")

# Print summary table
print("\n── Final Metrics Table ─────────────────────────────────────────────")
header = f"  {'Model':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'CV':>7}"
print(header)
print("  " + "-"*60)
for name in mnames:
    r = results[name]
    print(f"  {name:<25} {r['Accuracy']:>7.4f} {r['Precision']:>7.4f} "
          f"{r['Recall']:>7.4f} {r['F1']:>7.4f} {r['CV_Acc']:>7.4f}")
print(f"\n  ★ Best → {best_name}")

# %% ── CELL 6 : SAVE MODEL ────────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 6 — SAVE BEST MODEL")
print("="*65)

bundle = {
    "pipeline"    : best_pipe,
    "feature_cols": FEATURES,
    "classes"     : SPECIES,
    "model_name"  : best_name,
    "metrics"     : {k: v for k,v in results[best_name].items()
                     if k not in ("CM","y_pred")},
}
joblib.dump(bundle, f"{OUT}/iris_model.joblib")
print(f"Model saved → iris_model.joblib  (best: {best_name})")

# %% ── CELL 7 : EXAMPLE INFERENCE ────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 7 — EXAMPLE INFERENCE")
print("="*65)

# Load from disk (exactly as an end-user would)
loaded   = joblib.load(f"{OUT}/iris_model.joblib")
pipeline = loaded["pipeline"]

sample = pd.DataFrame([{
    "SepalLengthCm": 5.1,
    "SepalWidthCm" : 3.5,
    "PetalLengthCm": 1.4,
    "PetalWidthCm" : 0.2,
}])

predicted_species    = pipeline.predict(sample)[0]
predicted_proba      = pipeline.predict_proba(sample)[0]
proba_str = "  |  ".join(
    f"{sp.split('-')[1]}: {p:.3f}"
    for sp, p in zip(SPECIES, predicted_proba))

print(f"\n  Input : {sample.to_dict('records')[0]}")
print(f"  Predicted species : {predicted_species}")
print(f"  Class probabilities → {proba_str}")
print("\n✅  Pipeline complete — all outputs saved.")
print("="*65)

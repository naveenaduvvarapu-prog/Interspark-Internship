# Iris Species Classification — Model README

**Dataset:** [Kaggle — Iris Classification](https://www.kaggle.com/datasets/bhanupratapbiswas/iris-classification-dataset)  
**Best model:** Decision Tree (max_depth=4)  
**Test Accuracy:** 96.67% | **F1 (macro):** 0.9666 | **CV Accuracy:** 96.67%

---

## Files

| File | Description |
|---|---|
| `iris_notebook.py` | Full pipeline — EDA → training → evaluation (7 cells) |
| `iris_model.joblib` | Saved best-model pipeline (StandardScaler + Decision Tree) |
| `01_eda.png` | EDA charts — scatter plots, box plots, correlation heatmap |
| `02_model_evaluation.png` | Confusion matrices, metric comparison, decision tree plot |
| `README.md` | This file |

---

## Requirements

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib
```

---

## Run the full notebook

```bash
python iris_notebook.py
```

To use your own Kaggle CSV, replace the data-generation block in **Cell 1** with:

```python
df = pd.read_csv("Iris.csv")
```

---

## Inference — predict a new flower

```python
import joblib
import pandas as pd

# Load saved model
bundle   = joblib.load("iris_model.joblib")
pipeline = bundle["pipeline"]
classes  = bundle["classes"]   # ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Build input (all measurements in cm)
sample = pd.DataFrame([{
    "SepalLengthCm": 6.3,
    "SepalWidthCm" : 3.3,
    "PetalLengthCm": 6.0,
    "PetalWidthCm" : 2.5,
}])

# Predict
species       = pipeline.predict(sample)[0]
probabilities = pipeline.predict_proba(sample)[0]

print(f"Predicted species : {species}")
for cls, prob in zip(classes, probabilities):
    print(f"  {cls:<20} {prob:.3f}")
```

**Expected output:**
```
Predicted species : Iris-virginica
  Iris-setosa          0.000
  Iris-versicolor      0.000
  Iris-virginica       1.000
```

---

## Model Performance Summary

| Model | Accuracy | Precision | Recall | F1 | CV Acc |
|---|---|---|---|---|---|
| k-NN (k=5) | 93.33% | 94.44% | 93.33% | 93.27% | 97.33% |
| Logistic Regression | 93.33% | 93.33% | 93.33% | 93.33% | 95.33% |
| **Decision Tree ★** | **96.67%** | **96.97%** | **96.67%** | **96.66%** | **96.67%** |

### Key findings
- **Iris-setosa** is perfectly separable from the other two species using petal features alone.
- **PetalLengthCm** and **PetalWidthCm** are the most discriminative features (high correlation with class label).
- **SepalWidthCm** is the least informative feature in isolation.
- The Decision Tree achieves near-perfect accuracy with depth ≤ 4, making it both accurate and interpretable.

---

## Feature Schema

| Feature | Unit | Range |
|---|---|---|
| SepalLengthCm | cm | 4.3 – 7.9 |
| SepalWidthCm | cm | 2.0 – 4.4 |
| PetalLengthCm | cm | 1.0 – 6.9 |
| PetalWidthCm | cm | 0.1 – 2.5 |

**Target classes:** `Iris-setosa` · `Iris-versicolor` · `Iris-virginica`

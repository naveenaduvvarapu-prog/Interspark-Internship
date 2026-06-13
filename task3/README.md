# Titanic Survival Prediction

**Goal:** Predict passenger survival using engineered features and interpretable ML models.

---

## Project Structure

```
titanic_model/
├── best_model.pkl        # Saved best classifier (highest ROC-AUC)
├── label_encoders.pkl    # Fitted LabelEncoders for categorical features
├── feature_names.pkl     # Ordered feature list
└── metrics.json          # Final evaluation metrics for all models

titanic_notebook.py       # Full pipeline: EDA → Feature Eng → Train → Explain
titanic_dashboard.png     # Visual report (charts + metric cards)
README.md                 # This file
```

---

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

Then download the dataset from Kaggle and replace the data generation block in  
`titanic_notebook.py` with `df = pd.read_csv('train.csv')`.

---

## Feature Engineering Summary

| Feature | Strategy |
|---|---|
| **Title** | Extracted from Name; rare titles → `Rare` |
| **FamilySize** | `SibSp + Parch + 1` |
| **IsAlone** | Binary: FamilySize == 1 |
| **FamilyBand** | Binned: Alone / Small (2–4) / Large (5+) |
| **HasCabin** | Binary cabin presence (class proxy) |
| **Deck** | Cabin letter, or `U` if missing |
| **Age** | Imputed by `median(Title, Pclass)` → global median |
| **AgeBand** | Child / Teen / YoungAdult / Adult / Senior |
| **LogFare** | `log1p(Fare)` to reduce right skew |
| **Embarked** | Mode-imputed |

---

## Models Trained

- **Logistic Regression** — strong baseline, fully interpretable coefficients  
- **Random Forest** — ensemble, built-in feature importance  
- **Gradient Boosting** — sequential boosting, high accuracy  

Selection: **best ROC-AUC** on held-out test set (80/20 split, stratified).

---

## Key Results

| Model | CV Acc | Test Acc | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~0.747 | ~0.832 | **~0.890** |
| Random Forest | ~0.765 | ~0.832 | ~0.871 |
| Gradient Boosting | ~0.723 | ~0.765 | ~0.823 |

---

## Explainability

Two complementary methods are used (no SHAP required):

1. **Tree / Coefficient Importance** — built-in `.feature_importances_` or model weights  
2. **Permutation Importance** — model-agnostic; measures AUC drop when each feature is randomly shuffled

Top drivers: `Sex`, `Title`, `Pclass`, `LogFare`, `HasCabin`

---

## Inference Example

```python
from titanic_notebook import predict_passenger

result = predict_passenger({
    'Pclass': 3,
    'Name': 'Smith, Mr. John',
    'Sex': 'male',
    'Age': 28.0,
    'SibSp': 0,
    'Parch': 0,
    'Fare': 7.50,
    'Cabin': None,
    'Embarked': 'S'
})
# → {'survived': False, 'probability': 0.083}
```

**Input fields:** `Pclass`, `Name` (for Title extraction), `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Cabin` (or `None`), `Embarked`.

---

## Missing Value Strategy

| Column | % Missing | Strategy |
|---|---|---|
| Age | ~20% | Median per (Title, Pclass) group; global median fallback |
| Cabin | ~75% | Converted to binary `HasCabin`; letter → `Deck` ('U' if absent) |
| Embarked | ~0.2% | Mode imputation ('S') |

---

*Dataset source: [Kaggle — Titanic Survival Dataset](https://www.kaggle.com/datasets/bhanupratapbiswas/titanic-survival-datasets)*

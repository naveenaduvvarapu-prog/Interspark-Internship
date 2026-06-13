# =============================================================================
# TITANIC SURVIVAL PREDICTION — Full Notebook
# =============================================================================
# Run: python titanic_notebook.py
# Requirements: pandas numpy scikit-learn matplotlib joblib
# =============================================================================

# %% [1] Imports & Setup
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, joblib, json, os
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

# %% [2] Load / Generate Data
# ─────────────────────────────────────────────────────────────────────────────
# Replace the block below with:
#   df = pd.read_csv('train.csv')
# if you have the Kaggle CSV from:
#   https://www.kaggle.com/datasets/bhanupratapbiswas/titanic-survival-datasets
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)
n = 891
pclass   = np.random.choice([1,2,3], n, p=[0.24,0.21,0.55])
sex_raw  = np.random.choice(['male','female'], n, p=[0.65,0.35])
surv_p   = np.where(sex_raw=='female', 0.74, 0.19)
surv_p   = np.where(pclass==1, surv_p+0.15, surv_p)
surv_p   = np.where(pclass==3, surv_p-0.10, surv_p)
survived = (np.random.rand(n) < np.clip(surv_p,0,1)).astype(int)
age_vals = np.clip(np.random.normal(29,14,n), 0.5, 80)
age      = np.where(np.random.rand(n)<0.20, np.nan, age_vals)
sibsp    = np.random.choice([0,1,2,3,4,5], n, p=[0.60,0.23,0.09,0.04,0.02,0.02])
parch    = np.random.choice([0,1,2,3,4,5], n, p=[0.76,0.13,0.08,0.01,0.01,0.01])
fare     = np.array([np.random.exponential({1:85,2:22,3:13}[p]) for p in pclass])
cabin_c  = np.random.choice(['C85','B28','D56','E46','A10'], n)
has_cab  = np.random.rand(n) < 0.23
cabin_raw = np.where(has_cab, cabin_c, None).astype(object)
emb_c    = np.random.choice(['S','C','Q'], n, p=[0.72,0.19,0.09])
has_emb  = np.random.rand(n) < 0.02
embarked = np.where(has_emb, None, emb_c).astype(object)
title_pool_m = ['Mr','Dr','Rev','Col','Major','Master']
title_pool_f = ['Mrs','Miss','Lady']
titles   = np.where(sex_raw=='male',
               np.random.choice(title_pool_m, n, p=[0.87,0.04,0.04,0.03,0.01,0.01]),
               np.random.choice(title_pool_f, n, p=[0.55,0.43,0.02]))
names    = [f"Doe, {t}. John" for t in titles]
df = pd.DataFrame({
    'PassengerId': range(1,n+1), 'Survived': survived, 'Pclass': pclass,
    'Name': names, 'Sex': sex_raw, 'Age': age, 'SibSp': sibsp,
    'Parch': parch, 'Fare': fare, 'Cabin': cabin_raw, 'Embarked': embarked
})
print(f"Dataset: {df.shape}, Survival rate: {df.Survived.mean():.1%}")
print(f"Missing — Age: {df.Age.isna().sum()}, Cabin: {df.Cabin.isna().sum()}")

# %% [3] Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# • Title     — extracted from Name; rare titles merged into 'Rare'
# • FamilySize — SibSp + Parch + 1 (self)
# • IsAlone   — binary flag for solo passengers
# • FamilyBand — binned family size: Alone / Small / Large
# • HasCabin  — binary cabin presence (proxy for 1st-class deck)
# • Deck      — cabin letter or 'U' (unknown)
# • Age       — imputed by median(Title, Pclass) then global median
# • AgeBand   — binned age: Child/Teen/YoungAdult/Adult/Senior
# • LogFare   — log1p-transformed fare (right-skewed)
# • Embarked  — mode-imputed
# ─────────────────────────────────────────────────────────────────────────────

def engineer(df):
    d = df.copy()
    d['Title'] = d['Name'].str.extract(r',\s*([^\.]+)\.')
    rare = ['Rev','Col','Major','Lady','Sir','Countess','Capt','Don','Jonkheer','Dona']
    d['Title'] = d['Title'].replace(rare, 'Rare')
    d['Title'] = d['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})

    d['FamilySize'] = d['SibSp'] + d['Parch'] + 1
    d['IsAlone']    = (d['FamilySize'] == 1).astype(int)
    d['FamilyBand'] = pd.cut(d['FamilySize'], bins=[0,1,4,11],
                              labels=['Alone','Small','Large'])
    d['HasCabin']   = d['Cabin'].notna().astype(int)
    d['Deck']       = d['Cabin'].apply(lambda x: x[0] if isinstance(x, str) else 'U')

    d['Age'] = d.groupby(['Title','Pclass'])['Age'].transform(
        lambda x: x.fillna(x.median()))
    d['Age'] = d['Age'].fillna(d['Age'].median())
    d['AgeBand'] = pd.cut(d['Age'], bins=[0,12,18,35,60,100],
                           labels=['Child','Teen','YoungAdult','Adult','Senior'])

    d['LogFare']   = np.log1p(d['Fare'])
    d['Embarked']  = d['Embarked'].fillna(d['Embarked'].mode()[0])
    return d

df = engineer(df)

# %% [4] Encode & Build Feature Matrix
cat_cols  = ['Sex','Title','Embarked','FamilyBand','AgeBand','Deck']
encoders  = {}
for c in cat_cols:
    le = LabelEncoder()
    df[c+'_enc'] = le.fit_transform(df[c].astype(str))
    encoders[c]  = le

features  = ['Pclass','Sex_enc','Age','SibSp','Parch','LogFare','HasCabin',
             'FamilySize','IsAlone','Title_enc','Embarked_enc',
             'FamilyBand_enc','AgeBand_enc','Deck_enc']
X, y = df[features], df['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print(f"\nFeatures: {features}")
print(f"Train/Test split: {X_train.shape[0]} / {X_test.shape[0]}")

# %% [5] Train & Evaluate Models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=300, max_depth=6,
                                                   min_samples_leaf=4, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                       learning_rate=0.05, random_state=42),
}
results = {}
for name, model in models.items():
    cv_acc  = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_auc  = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:,1]
    rep     = classification_report(y_test, y_pred, output_dict=True)
    results[name] = dict(model=model, cv_acc=cv_acc.mean(), cv_std=cv_acc.std(),
                          test_acc=accuracy_score(y_test, y_pred),
                          test_auc=roc_auc_score(y_test, y_prob),
                          y_pred=y_pred, y_prob=y_prob, report=rep)
    print(f"{name:25s}  CV={cv_acc.mean():.4f}±{cv_acc.std():.4f}  "
          f"Acc={accuracy_score(y_test, y_pred):.4f}  "
          f"AUC={roc_auc_score(y_test, y_prob):.4f}")

best_name  = max(results, key=lambda k: results[k]['test_auc'])
best_model = results[best_name]['model']
print(f"\nBest model: {best_name}")

# %% [6] Feature Importance & Explainability
if hasattr(best_model, 'feature_importances_'):
    imp_vals = best_model.feature_importances_
else:
    from sklearn.inspection import permutation_importance as pi
    r        = pi(best_model, X_test, y_test, n_repeats=15, random_state=42)
    imp_vals = r.importances_mean

feat_imp = pd.Series(imp_vals, index=features).sort_values(ascending=False)
perm     = permutation_importance(best_model, X_test, y_test,
                                   n_repeats=15, random_state=42, scoring='roc_auc')
perm_imp_s = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)

print("\nTop-8 Features (Tree/Model Importance):")
print(feat_imp.head(8).to_string())
print("\nTop-8 Features (Permutation Importance — AUC drop):")
print(perm_imp_s.head(8).to_string())

# %% [7] Save Model & Metrics
os.makedirs('titanic_model', exist_ok=True)
joblib.dump(best_model, 'titanic_model/best_model.pkl')
joblib.dump(encoders,   'titanic_model/label_encoders.pkl')
joblib.dump(features,   'titanic_model/feature_names.pkl')

metrics_out = {
    name: dict(cv_acc=round(r['cv_acc'],4), cv_std=round(r['cv_std'],4),
               test_acc=round(r['test_acc'],4), test_auc=round(r['test_auc'],4),
               precision=round(r['report']['weighted avg']['precision'],4),
               recall=round(r['report']['weighted avg']['recall'],4),
               f1=round(r['report']['weighted avg']['f1-score'],4))
    for name, r in results.items()
}
with open('titanic_model/metrics.json','w') as f:
    json.dump(metrics_out, f, indent=2)

print(f"\nModel saved → titanic_model/")

# %% [8] Inference Example
# ─────────────────────────────────────────────────────────────────────────────
def predict_passenger(record: dict) -> dict:
    """
    record keys: Pclass, Name, Sex, Age, SibSp, Parch, Fare, Cabin, Embarked
    Returns: {'survived': bool, 'probability': float}
    """
    model    = joblib.load('titanic_model/best_model.pkl')
    encs     = joblib.load('titanic_model/label_encoders.pkl')
    feat_nms = joblib.load('titanic_model/feature_names.pkl')

    row = pd.DataFrame([{**record, 'PassengerId': 0, 'Survived': -1}])
    row = engineer(row)
    for c, le in encs.items():
        val = str(row[c].iloc[0])
        if val not in le.classes_:
            val = le.classes_[0]
        row[c+'_enc'] = le.transform([val])[0]

    prob = model.predict_proba(row[feat_nms])[0][1]
    return {'survived': bool(prob >= 0.5), 'probability': round(prob, 4)}

# Example passengers
passengers = [
    {'Pclass':1,'Name':'Fortune, Mrs. Mark','Sex':'female','Age':24,'SibSp':1,
     'Parch':0,'Fare':263.0,'Cabin':'C23','Embarked':'S'},
    {'Pclass':3,'Name':'Goodwin, Mr. Charles','Sex':'male','Age':14,'SibSp':5,
     'Parch':2,'Fare':46.9,'Cabin':None,'Embarked':'S'},
    {'Pclass':2,'Name':'Niqette, Mr. Julien','Sex':'male','Age':30,'SibSp':0,
     'Parch':0,'Fare':13.0,'Cabin':None,'Embarked':'C'},
]
print("\nInference examples:")
for p in passengers:
    result = predict_passenger(p)
    icon   = '✓ Survived' if result['survived'] else '✗ Did not survive'
    print(f"  {p['Name'][:30]:<30}  {icon}  P={result['probability']:.3f}")

print("\nDone.")

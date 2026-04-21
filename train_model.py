import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# NEW - always works regardless of where you run from
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "kidney_disease_train.csv"))
df['classification'] = df['classification'].str.strip().map({'ckd': 1, 'notckd': 0})
for col in ['wc', 'rc']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

FEATURES = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc',
            'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']

FEATURE_LABELS = {
    'age':'Age','bp':'Blood Pressure','sg':'Specific Gravity',
    'al':'Albumin','su':'Sugar','bgr':'Blood Glucose (Random)',
    'bu':'Blood Urea','sc':'Serum Creatinine','sod':'Sodium',
    'pot':'Potassium','hemo':'Hemoglobin','pcv':'Packed Cell Volume',
    'wc':'White Cell Count','rc':'Red Cell Count'
}

NORMAL_RANGES = {
    'age':(0,100),'bp':(60,90),'sg':(1.010,1.025),
    'al':(0,0),'su':(0,0),'bgr':(70,140),
    'bu':(7,20),'sc':(0.6,1.2),'sod':(136,145),
    'pot':(3.5,5.0),'hemo':(12,17),'pcv':(36,50),
    'wc':(4500,11000),'rc':(4.5,5.5)
}

X = df[FEATURES]
y = df['classification']

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y)

model_defs = {
    "Random Forest":     RandomForestClassifier(n_estimators=150, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
    "SVM (RBF Kernel)":  make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=42))
}

models = {}
model_metrics = {}

print("=" * 55)
print("      RenalSense AI - Model Training Report")
print("=" * 55)

for name, clf in model_defs.items():
    clf.fit(X_train, y_train)
    cv_scores = cross_val_score(clf, X_imputed, y, cv=5)
    val_acc   = accuracy_score(y_val, clf.predict(X_val))
    cm        = confusion_matrix(y_val, clf.predict(X_val))
    models[name] = clf
    model_metrics[name] = {
        "cv_accuracy":      round(cv_scores.mean()*100, 2),
        "cv_std":           round(cv_scores.std()*100,  2),
        "val_accuracy":     round(val_acc*100,           2),
        "confusion_matrix": cm.tolist()
    }
    print(f"\n{name}")
    print(f"  5-Fold CV Accuracy : {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")
    print(f"  Validation Accuracy: {val_acc*100:.2f}%")

# Feature importance from Random Forest
rf_model = models["Random Forest"]
importances = dict(zip(FEATURES, rf_model.feature_importances_))

print("\nTop Feature Importances (Random Forest):")
for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
    print(f"  {FEATURE_LABELS[feat]:25s} {imp:.4f}")

payload = {
    "models":         models,
    "model_metrics":  model_metrics,
    "imputer":        imputer,
    "features":       FEATURES,
    "feature_labels": FEATURE_LABELS,
    "normal_ranges":  NORMAL_RANGES,
    "importances":    importances
}

with open("kidney_model.pkl", "wb") as f:
    pickle.dump(payload, f)

print("\n✅ kidney_model.pkl saved successfully!")

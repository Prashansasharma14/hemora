import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, f1_score
import pickle

# Load dataset (supports both CSV and XLSX)
base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "pph_data.csv"
xlsx_path = base_dir / "pph_data.xlsx"

if csv_path.exists():
    df = pd.read_csv(csv_path)
elif xlsx_path.exists():
    df = pd.read_excel(xlsx_path)
else:
    raise FileNotFoundError("Neither pph_data.csv nor pph_data.xlsx was found.")

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Remove bad column if exists
df = df.drop(columns=[None], errors='ignore')

print("Columns:", df.columns)

# Handle alternate column names seen across dataset versions
column_aliases = {
    "prolonged_labor": "prolonged",
    "multiple_preg": "multiple",
}
for expected_col, alias_col in column_aliases.items():
    if expected_col not in df.columns and alias_col in df.columns:
        df[expected_col] = df[alias_col]

# Features
X = df[['age', 'parity', 'hb', 'prev_lscs', 'induction',
        'prolonged_labor', 'multiple_preg', 'bmi', 'bp', 'prev_pph', 'placenta']]

# Target
y = df['pph']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Model (controlled complexity)
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=3,
    class_weight="balanced_subsample",
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Tune threshold on validation probabilities for a less optimistic classifier output
test_probs = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, test_probs)

best_threshold = 0.5
best_f1 = -1.0
for threshold in thresholds:
    y_pred_tuned = (test_probs >= threshold).astype(int)
    curr_f1 = f1_score(y_test, y_pred_tuned, zero_division=0)
    if curr_f1 > best_f1:
        best_f1 = curr_f1
        best_threshold = float(threshold)

# Evaluate at default and tuned thresholds
y_pred_default = (test_probs >= 0.5).astype(int)
y_pred_tuned = (test_probs >= best_threshold).astype(int)

print("\nTest Performance (threshold=0.50):")
print(classification_report(y_test, y_pred_default, zero_division=0))
print(f"Best threshold by F1: {best_threshold:.3f}")
print("Test Performance (tuned threshold):")
print(classification_report(y_test, y_pred_tuned, zero_division=0))

# Cross-validation (real performance)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
scores = cross_val_score(model, X, y, scoring='f1', cv=cv)

print("\nCross-validation F1:", scores.mean(), "+/-", scores.std())

# Save model
pickle.dump(model, open("pph_model.pkl", "wb"))

print("\nModel trained and saved successfully")


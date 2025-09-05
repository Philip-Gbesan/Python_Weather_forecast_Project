import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# =========================
# Paths
# =========================
DATA_PATH = "data/weatherAUS.csv"
MODELS_DIR = "models"

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# =========================
# Load dataset
# =========================
print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# =========================
# Preprocessing
# =========================

# Keep relevant features
features = [
    "MinTemp", "MaxTemp", "Rainfall",
    "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm",
    "Temp9am", "Temp3pm"
]
target = "RainTomorrow"

df = df[features + [target]]

# Handle missing values
for col in features:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Encode target variable (Yes/No → 1/0)
df = df.dropna(subset=[target])
le = LabelEncoder()
df[target] = le.fit_transform(df[target])  # Yes = 1, No = 0

# Split data
X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

# =========================
# Train models
# =========================
models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "svm": SVC(probability=True, kernel="rbf", random_state=42)
}

accuracies = {}

for name, model in models.items():
    print(f"🔄 Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    accuracies[name] = round(acc, 2)
    joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))

# Save accuracies
joblib.dump(accuracies, os.path.join(MODELS_DIR, "accuracies.pkl"))

print("✅ Training complete. Models saved in /models")
print("Model Accuracies:", accuracies)

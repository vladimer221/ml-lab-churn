import json
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
import mlflow.sklearn

# ====== ЗАГРУЗКА ДАННЫХ ======
train = pd.read_csv("data/splits/train.csv")
val = pd.read_csv("data/splits/val.csv")

X_train = train.drop(columns=["Churn"])
y_train = train["Churn"]
X_val = val.drop(columns=["Churn"])
y_val = val["Churn"]

print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

# ====== BASELINE С PIPELINE ======
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42
    ))
])

print("Training Logistic Regression Baseline")
model.fit(X_train, y_train)

# ====== ПРЕДСКАЗАНИЯ ======
proba = model.predict_proba(X_val)[:, 1]
pred = (proba >= 0.5).astype(int)

# ====== МЕТРИКИ ======
metrics = {
    "model": "logreg_baseline_scaled",
    "f1": float(f1_score(y_val, pred)),
    "roc_auc": float(roc_auc_score(y_val, proba)),
    "train_score": float(model.score(X_train, y_train))
}

# ====== MLflow ======
mlflow.set_experiment("telco_churn")
with mlflow.start_run(run_name="logreg_baseline_scaled"):
    mlflow.log_param("model", "logreg")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("max_iter", 3000)
    mlflow.log_metric("f1", metrics["f1"])
    mlflow.log_metric("roc_auc", metrics["roc_auc"])
    mlflow.log_metric("train_score", metrics["train_score"])
    mlflow.sklearn.log_model(model, "model")

# ====== СОХРАНЕНИЕ ======
Path("artifacts").mkdir(exist_ok=True)

Path("artifacts/baseline_metrics.json").write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8"
)

joblib.dump(model, "artifacts/baseline_model.pkl")

pd.DataFrame({
    "pred": pred,
    "proba": proba
}).to_csv("artifacts/baseline_pred_val.csv", index=False)

print("Baseline training completed")
print(f"F1-score:     {metrics['f1']:.4f}")
print(f"ROC-AUC:      {metrics['roc_auc']:.4f}")
print(f"Train score:  {metrics['train_score']:.4f}")
print("Model saved → artifacts/baseline_model.pkl")
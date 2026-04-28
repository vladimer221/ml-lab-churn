import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path

print("Loading model and data...")

model = joblib.load("artifacts/baseline_model.pkl")
df = pd.read_csv("data/splits/train.csv")
X = df.drop(columns=["Churn"])

# Берём подвыборку для скорости
X_sample = X.sample(100, random_state=42)

print("Extracting pipeline steps...")

scaler = model.named_steps["scaler"]
logreg = model.named_steps["logreg"]

# SHAP для линейной модели нужно считать на уже scaled-признаках
X_background = pd.DataFrame(
    scaler.transform(X),
    columns=X.columns
)

X_sample_scaled = pd.DataFrame(
    scaler.transform(X_sample),
    columns=X.columns
)

print("Creating SHAP LinearExplainer...")

explainer = shap.LinearExplainer(logreg, X_background)
shap_values = explainer.shap_values(X_sample_scaled)

if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = np.asarray(shap_values, dtype=float)

Path("artifacts").mkdir(exist_ok=True)

print("Saving plots...")

# summary plot
shap.summary_plot(shap_values, X_sample_scaled, show=False)
plt.tight_layout()
plt.savefig("artifacts/shap_summary.png", dpi=200, bbox_inches="tight")
plt.close()

# bar plot
shap.summary_plot(
    shap_values,
    X_sample_scaled,
    plot_type="bar",
    show=False
)

plt.xlabel("Среднее |SHAP значение| (влияние признака на модель)", fontsize=10)

plt.tight_layout()
plt.savefig(
    "artifacts/shap_bar.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("SHAP completed")
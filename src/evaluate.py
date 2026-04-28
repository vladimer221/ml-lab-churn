import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

VAL_PATH = Path("data/splits/val.csv")
TEST_PATH = Path("data/splits/test.csv")
BASELINE_MODEL_PATH = ARTIFACTS_DIR / "baseline_model.pkl"
NN_MODEL_PATH = ARTIFACTS_DIR / "mlp_model.pt"
NN_SCALER_PATH = ARTIFACTS_DIR / "nn_scaler.pkl"


class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def save_classification_artifacts(prefix: str, y_true, y_pred, proba, X_df):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    roc_auc = float(roc_auc_score(y_true, proba))

    (ARTIFACTS_DIR / f"eval_report_{prefix}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (ARTIFACTS_DIR / f"confusion_matrix_{prefix}.json").write_text(
        json.dumps(cm, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (ARTIFACTS_DIR / f"roc_auc_{prefix}.json").write_text(
        json.dumps({"roc_auc": roc_auc}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    mistakes = X_df.copy()
    mistakes["true"] = y_true
    mistakes["pred"] = y_pred
    mistakes["proba"] = proba

    wrong = mistakes[mistakes["true"] != mistakes["pred"]]
    wrong.head(20).to_csv(ARTIFACTS_DIR / f"errors_top20_{prefix}.csv", index=False)
    wrong.to_csv(ARTIFACTS_DIR / f"errors_all_{prefix}.csv", index=False)

    if len(wrong) > 0:
        feature_profile = (
            wrong.drop(columns=["true", "pred", "proba"])
            .mean(numeric_only=True)
            .sort_values(ascending=False)
        )
        feature_profile.head(10).to_csv(
            ARTIFACTS_DIR / f"top_features_in_errors_{prefix}.csv"
        )

    return {
        "accuracy": float(report["accuracy"]),
        "f1": float(report["1"]["f1-score"]),
        "roc_auc": roc_auc,
        "mistakes": int(len(wrong)),
    }


print("Loading validation and test data...")

val = pd.read_csv(VAL_PATH)
test = pd.read_csv(TEST_PATH)

X_val = val.drop(columns=["Churn"])
y_val = val["Churn"].astype(int).values

X_test = test.drop(columns=["Churn"])
y_test = test["Churn"].astype(int).values

results = {}

# ====== BASELINE ======
print("Evaluating baseline model...")

if not BASELINE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Baseline model not found at {BASELINE_MODEL_PATH}. "
        "Run src/train_baseline.py first."
    )

baseline_model = joblib.load(BASELINE_MODEL_PATH)

baseline_val_proba = baseline_model.predict_proba(X_val)[:, 1]
baseline_val_pred = (baseline_val_proba >= 0.5).astype(int)

pd.DataFrame(
    {
        "pred": baseline_val_pred,
        "proba": baseline_val_proba,
    }
).to_csv(ARTIFACTS_DIR / "pred_val.csv", index=False)

results["baseline_val"] = save_classification_artifacts(
    "baseline",
    y_val,
    baseline_val_pred,
    baseline_val_proba,
    X_val,
)

baseline_test_proba = baseline_model.predict_proba(X_test)[:, 1]
baseline_test_pred = (baseline_test_proba >= 0.5).astype(int)

results["baseline_test"] = {
    "roc_auc": float(roc_auc_score(y_test, baseline_test_proba)),
    "f1": float(f1_score(y_test, baseline_test_pred)),
}

# ====== NEURAL NETWORK ======
if NN_MODEL_PATH.exists() and NN_SCALER_PATH.exists():
    print("Evaluating neural network...")

    nn_ckpt = torch.load(NN_MODEL_PATH, map_location="cpu")
    nn_scaler = joblib.load(NN_SCALER_PATH)

    if isinstance(nn_ckpt, dict) and "state_dict" in nn_ckpt:
        input_dim = int(nn_ckpt["input_dim"])
        state_dict = nn_ckpt["state_dict"]
    else:
        raise RuntimeError(
            "Unexpected NN checkpoint format. "
            "Expected a dict with keys: state_dict, input_dim."
        )

    nn_model = MLP(input_dim=input_dim)
    nn_model.load_state_dict(state_dict)
    nn_model.eval()

    X_val_nn = nn_scaler.transform(X_val.values)
    X_test_nn = nn_scaler.transform(X_test.values)

    X_val_nn_tensor = torch.tensor(X_val_nn, dtype=torch.float32)
    X_test_nn_tensor = torch.tensor(X_test_nn, dtype=torch.float32)

    with torch.no_grad():
        val_nn_proba = torch.sigmoid(nn_model(X_val_nn_tensor)).cpu().numpy()
        test_nn_proba = torch.sigmoid(nn_model(X_test_nn_tensor)).cpu().numpy()

    val_nn_pred = (val_nn_proba >= 0.5).astype(int)
    test_nn_pred = (test_nn_proba >= 0.5).astype(int)

    pd.DataFrame(
        {
            "pred": val_nn_pred,
            "proba": val_nn_proba,
        }
    ).to_csv(ARTIFACTS_DIR / "pred_val_nn.csv", index=False)

    results["nn_val"] = save_classification_artifacts(
        "nn",
        y_val,
        val_nn_pred,
        val_nn_proba,
        X_val,
    )

    results["nn_test"] = {
        "roc_auc": float(roc_auc_score(y_test, test_nn_proba)),
        "f1": float(f1_score(y_test, test_nn_pred)),
    }
else:
    print("NN artifacts not found. Skipping NN evaluation.")

(ARTIFACTS_DIR / "evaluation_summary.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\nEvaluation completed")
print(f"Baseline validation Accuracy: {results['baseline_val']['accuracy']:.4f}")
print(f"Baseline validation F1:       {results['baseline_val']['f1']:.4f}")
print(f"Baseline validation ROC-AUC:  {results['baseline_val']['roc_auc']:.4f}")
print(f"Baseline validation mistakes: {results['baseline_val']['mistakes']}")
print(f"Baseline test ROC-AUC:        {results['baseline_test']['roc_auc']:.4f}")
print(f"Baseline test F1:             {results['baseline_test']['f1']:.4f}")

if "nn_val" in results:
    print(f"NN validation F1:             {results['nn_val']['f1']:.4f}")
    print(f"NN validation ROC-AUC:        {results['nn_val']['roc_auc']:.4f}")
    print(f"NN test ROC-AUC:              {results['nn_test']['roc_auc']:.4f}")
    print(f"NN test F1:                   {results['nn_test']['f1']:.4f}")
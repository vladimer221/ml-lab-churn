import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    import mlflow
    import mlflow.pytorch

    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

TRAIN_PATH = Path("data/splits/train.csv")
VAL_PATH = Path("data/splits/val.csv")

train = pd.read_csv(TRAIN_PATH)
val = pd.read_csv(VAL_PATH)

X_train_df = train.drop(columns=["Churn"])
y_train = train["Churn"].astype(int).values

X_val_df = val.drop(columns=["Churn"])
y_val = val["Churn"].astype(int).values

feature_names = list(X_train_df.columns)

print(f"Train shape: {X_train_df.shape}, Val shape: {X_val_df.shape}")

# ====== SCALING ======
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_df.values)
X_val = scaler.transform(X_val_df.values)

joblib.dump(scaler, ARTIFACTS_DIR / "nn_scaler.pkl")

# ====== TENSORS ======
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# ====== MODEL ======
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


model = MLP(input_dim=X_train.shape[1])

# ====== LOSS / OPTIMIZER ======
pos = float(y_train.sum())
neg = float(len(y_train) - y_train.sum())
pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# ====== TRAINING ======
epochs = 80
best_state = None
best_val_f1 = -1.0
history = []

print("Training PyTorch Neural Network...")

for epoch in range(1, epochs + 1):
    model.train()
    train_losses = []

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_probs = []
    with torch.no_grad():
        for batch_x, _ in val_loader:
            logits = model(batch_x)
            probs = torch.sigmoid(logits)
            val_probs.append(probs.cpu().numpy())

    val_probs = np.concatenate(val_probs)
    val_pred = (val_probs >= 0.5).astype(int)

    val_f1 = f1_score(y_val, val_pred)
    val_roc_auc = roc_auc_score(y_val, val_probs)
    avg_train_loss = float(np.mean(train_losses))

    history.append(
        {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_f1": float(val_f1),
            "val_roc_auc": float(val_roc_auc),
        }
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = copy.deepcopy(model.state_dict())

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"loss={avg_train_loss:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"val_roc_auc={val_roc_auc:.4f}"
        )

# ====== RESTORE BEST MODEL ======
if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
with torch.no_grad():
    val_logits = model(X_val_tensor)
    val_proba = torch.sigmoid(val_logits).cpu().numpy()

val_pred = (val_proba >= 0.5).astype(int)

metrics = {
    "model": "pytorch_mlp",
    "f1": float(f1_score(y_val, val_pred)),
    "roc_auc": float(roc_auc_score(y_val, val_proba)),
    "epochs": epochs,
    "batch_size": 64,
    "learning_rate": 5e-4,
    "input_dim": int(X_train.shape[1]),
    "feature_count": int(len(feature_names)),
}

# ====== SAVE ARTIFACTS ======
torch.save(
    {
        "state_dict": model.state_dict(),
        "input_dim": int(X_train.shape[1]),
        "feature_names": feature_names,
        "seed": SEED,
    },
    ARTIFACTS_DIR / "mlp_model.pt",
)

(ARTIFACTS_DIR / "nn_metrics.json").write_text(
    json.dumps(metrics, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

(ARTIFACTS_DIR / "nn_training_history.json").write_text(
    json.dumps(history, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

pd.DataFrame(
    {
        "pred": val_pred,
        "proba": val_proba,
    }
).to_csv(ARTIFACTS_DIR / "pred_val_nn.csv", index=False)

# ====== OPTIONAL MLFLOW LOGGING ======
if MLFLOW_AVAILABLE:
    try:
        mlflow.set_experiment("telco_churn")
        with mlflow.start_run(run_name="pytorch_mlp"):
            mlflow.log_param("model", "pytorch_mlp")
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", 64)
            mlflow.log_param("learning_rate", 5e-4)
            mlflow.log_metric("f1", metrics["f1"])
            mlflow.log_metric("roc_auc", metrics["roc_auc"])
    except Exception as e:
        print(f"MLflow logging skipped: {e}")

print("Neural Network training completed!")
print(f"F1-score: {metrics['f1']:.4f}")
print(f"ROC-AUC:  {metrics['roc_auc']:.4f}")
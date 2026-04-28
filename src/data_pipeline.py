from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "raw" / "telco_churn.csv"
SPLIT_DIR = DATA_DIR / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW_PATH)

print("Shape:", df.shape)
print("\nTarget distribution:\n", df['Churn'].value_counts(normalize=True))

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

df = df.drop(columns=['customerID'])

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("After preprocessing shape:", df.shape)

target = "Churn"
X = df.drop(columns=[target])
y = df[target]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Сохраняем
train_df = X_train.assign(Churn=y_train.values)
val_df = X_val.assign(Churn=y_val.values)
test_df = X_test.assign(Churn=y_test.values)

train_df.to_csv(SPLIT_DIR / "train.csv", index=False)
val_df.to_csv(SPLIT_DIR / "val.csv", index=False)
test_df.to_csv(SPLIT_DIR / "test.csv", index=False)

print(" Splits saved to data/splits/")
print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
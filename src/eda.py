from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = Path("data/raw/telco_churn.csv")
OUTPUT_DIR = Path("artifacts/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)

# ====== RAW MISSING VALUES ======
missing_raw = df.isna().sum().sort_values(ascending=False)
missing_raw = missing_raw[missing_raw > 0]

print("\nMissing values (raw):\n", missing_raw)

missing_raw.to_csv(OUTPUT_DIR / "missing_raw.csv")

# ====== TARGET DISTRIBUTION ======
plt.figure(figsize=(6, 4))
df["Churn"].value_counts().plot(kind="bar")
plt.title("Target distribution (Churn)")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "target_distribution.png", dpi=200)
plt.close()

# ====== TYPE CLEANUP ======
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn_bin"] = df["Churn"].map({"No": 0, "Yes": 1})

# ====== MISSING AFTER CLEANUP ======
missing_after = df.isna().sum().sort_values(ascending=False)
missing_after = missing_after[missing_after > 0]
missing_after.to_csv(OUTPUT_DIR / "missing_after_cleanup.csv")

# ====== NUMERIC DISTRIBUTIONS ======
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Distribution: {col}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"dist_{col}.png", dpi=200)
    plt.close()

# ====== BOXPLOTS ======
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot: {col}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"box_{col}.png", dpi=200)
    plt.close()

# ====== CATEGORICAL DISTRIBUTIONS ======
categorical_cols = [
    col
    for col in df.columns
    if col not in numeric_cols and col not in ["Churn"]
]

important_categoricals = [
    col for col in ["Contract", "PaymentMethod", "InternetService", "gender"] if col in categorical_cols
]

for col in important_categoricals:
    plt.figure(figsize=(8, 4))
    df[col].value_counts().plot(kind="bar")
    plt.title(f"Distribution: {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"cat_{col}.png", dpi=200)
    plt.close()

# ====== CORRELATION ======
numeric_df = df.select_dtypes(include=["int64", "float64"]).copy()

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation.png", dpi=200)
plt.close()

# ====== SIMPLE SUMMARY ======
summary = pd.DataFrame(
    {
        "column": numeric_df.columns,
        "mean": numeric_df.mean().values,
        "std": numeric_df.std().values,
        "min": numeric_df.min().values,
        "max": numeric_df.max().values,
    }
)
summary.to_csv(OUTPUT_DIR / "numeric_summary.csv", index=False)

print("EDA completed. Files saved in artifacts/eda/")
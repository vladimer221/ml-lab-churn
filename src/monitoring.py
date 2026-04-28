import json
from pathlib import Path
import pandas as pd

log_path = "artifacts/prediction_logs.csv"

if Path(log_path).exists():
    logs = pd.read_csv(log_path)

    summary = {
        "count": int(len(logs)),
        "p95_latency_ms": float(logs["latency_ms"].quantile(0.95)),
        "avg_score": float(logs["score"].mean()),
        "max_score": float(logs["score"].max()),
        "min_score": float(logs["score"].min())
    }

    Path("docs").mkdir(exist_ok=True)
    Path("docs/monitoring_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(" Monitoring summary saved:")
    print(summary)
else:
    print(" No logs found. Call /predict first")
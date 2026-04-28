from pathlib import Path
import json

problem = {
    "business_goal": "Снизить отток клиентов телеком-компании, своевременно предлагая меры удержания активным клиентам",
    "ml_task_type": "binary_classification",
    "target_name": "Churn",
    "primary_metric": "f1",
    "secondary_metric": "roc_auc",
    "constraints": ["время предсказания < 50ms", "модель должна быть интерпретируемой"],
    "risks": ["сильный дисбаланс классов (~27% churn)", "возможный data leakage по tenure", "concept drift со временем"]
}

Path("docs").mkdir(exist_ok=True)
Path("docs/problem_statement.json").write_text(
    json.dumps(problem, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print("Problem statement saved to docs/problem_statement.json")
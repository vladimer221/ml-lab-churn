 Telco Customer Churn Prediction

Проект посвящён задаче прогнозирования оттока клиентов телеком-компании на основе исторических данных.
Реализован полный ML pipeline: от анализа данных до деплоя модели в виде API.



 Цель проекта

Снижение оттока клиентов за счёт предсказания вероятности ухода клиента (churn) и возможности дальнейшего применения retention-стратегий.



 Задача машинного обучения

 Тип задачи: бинарная классификация
 Целевая переменная: `Churn`
 Основная метрика: F1-score
 Дополнительная метрика: ROC-AUC

 Структура проекта


ml-lab-churn/
|
|- data/
|   -raw/
|   |- splits/
|
|- src/
|   |- data_pipeline.py
|   |- eda.py
|   |- train_baseline.py
|   |- train_nn.py
|   |- evaluate.py
|   |- explain.py
|   |- monitoring.py
|
|- app/
|   |- main.py
|
|- artifacts/
|- docs/
|- Dockerfile
|- requirements.txt
|- README.md

Этапы проекта

 1. EDA (Exploratory Data Analysis)

 анализ пропусков
 распределения признаков
 корреляции
 баланс классов

 2. Предобработка данных

 обработка `TotalCharges`
 кодирование категориальных признаков
 разбиение на train / val / test



 3. Baseline модель

 Logistic Regression
 StandardScaler
 учёт дисбаланса классов (`class_weight="balanced"`)



 4. Нейросеть (PyTorch)

 MLP архитектура
 BatchNorm + Dropout
 оптимизация через Adam
 подбор по F1



 5. Оценка моделей

 F1-score
 ROC-AUC
 confusion matrix
 анализ ошибок



 6. Explainability

 SHAP (LinearExplainer)
 анализ важности признаков



 7. Деплой

Реализован API на FastAPI:

 `GET /health` — проверка сервиса
 `POST /predict` — предсказание

Пример запроса:

json
{
  "features": {
    "tenure": 1,
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }
}
Ответ:
json
{
  "churn_probability": 0.67,
  "churn": 1,
  "latency_ms": 4.0
}

 8. Мониторинг
 логирование предсказаний
 latency
 распределение score
 подготовка к retrain

Как запустить проект

 1. Установка зависимостей

pip install -r requirements.txt

 2. Подготовка данных

python src/data_pipeline.py

 3. Обучение baseline

python src/train_baseline.py

 4. Обучение нейросети

python src/train_nn.py

 5. Оценка моделей

python src/evaluate.py

 6. SHAP анализ


python src/explain.py

 7. Запуск API


uvicorn app.main:app --reload

Swagger UI:

http://127.0.0.1:8000/docs

 Docker

docker build -t churn-api .
docker run -p 8000:8000 churn-api




 📈 Результаты

 Baseline (LogReg):

   F1 ≈ 0.62
   ROC-AUC ≈ 0.84

 Neural Network:

   улучшение F1
   сопоставимый ROC-AUC


 дисбаланс классов
 возможный data leakage
 concept drift


 подбор гиперпараметров
 более сложные модели (XGBoost, LightGBM)
 автоматический retrain
 продвинутый мониторинг


 Python
 Pandas / NumPy
 Scikit-learn
 PyTorch
 FastAPI
 MLflow
 SHAP

# -*- coding: utf-8 -*-
"""Model_Solar_Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QwztlbPQ9zIo479_oKpQM1x7VV-os8aV
"""

from google.colab import drive
drive.mount('/content/drive')

cd /content/drive/MyDrive/Capstone/data

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor
import joblib

df = pd.read_csv("processed_solar_data.csv")

features = ["AirTemperature", "RelativeHumidity", "WindSpeed", "hour", "kWp"]
X = df[features]
y = df["SolarGeneration"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.1]
}

model = GridSearchCV(
    XGBRegressor(objective='reg:squarederror', random_state=42),
    param_grid,
    cv=3,
    scoring="neg_mean_squared_error"
)

model.fit(X_train_scaled, y_train)
best_model = model.best_estimator_

import numpy as np

y_pred = best_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# 4. Оценка производительности
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Прогнозирование
y_pred = best_model.predict(X_test_scaled)

# Вычисление метрик
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5  # Ручное вычисление корня
r2 = r2_score(y_test, y_pred)

print(f"""
Метрики модели:
- MAE: {mean_absolute_error(y_test, y_pred):.2f}
- RMSE: {rmse:.2f}
- R2: {r2:.2f}
""")

# Аномалии
anomaly_scores = iso_forest.decision_function(X_test_scaled)
print(f"Обнаружено аномалий: {(anomaly_scores < 0).sum()} ({(anomaly_scores < 0).mean()*100:.1f}%)")

anomaly_detector = IsolationForest(
    n_estimators=100, contamination=0.05, random_state=42
)
anomaly_detector.fit(X_train_scaled)

joblib.dump(best_model, "solar_predictor.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(anomaly_detector, "anomaly_detector.pkl")
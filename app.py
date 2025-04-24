import streamlit as st
import pandas as pd
import joblib

model = joblib.load("solar_predictor.pkl")
scaler = joblib.load("scaler.pkl")
anomaly_detector = joblib.load("anomaly_detector.pkl")

st.title("Прогноз генерации солнечной энергии и выявление аномалий")

st.write("""
Это приложение предсказывает выработку солнечной энергии и определяет, 
насколько введённые погодные условия являются типичными или нет.
""")

st.header("Введите параметры")

col1, col2 = st.columns(2)
with col1:
    hour = st.slider("Час суток", 0, 23, 12)
    temp = st.number_input("Температура (°C)", value=25.0)
    kwp = st.number_input("Мощность установки (kWp)", min_value=1.0, value=100.0)

with col2:
    hum = st.number_input("Влажность (%)", min_value=0.0, max_value=100.0, value=50.0)
    wind = st.number_input("Скорость ветра (м/с)", min_value=0.0, value=2.0)

if st.button("Прогнозировать"):
    X = pd.DataFrame([[temp, hum, wind, hour, kwp]],
                     columns=["AirTemperature", "RelativeHumidity", "WindSpeed", "hour", "kWp"])
    
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    anomaly = anomaly_detector.predict(X_scaled)[0]

    st.header("Прогноз")

    st.write(f"Предсказанная генерация: **{prediction:.2f} кВт·ч**")

    st.header("Анализ условий")

    if anomaly == -1:
        st.error("Внимание: условия считаются аномальными.")
        st.markdown("""
        Аномалия означает, что введённые параметры (например, необычно высокая влажность,
        низкая температура или редкое сочетание факторов) **редко встречались в обучающей выборке**.
        В таких случаях модель может давать менее точный прогноз.
        """)
    else:
        st.success("Условия в пределах обучающей выборки.")
        st.write("Прогноз считается надёжным.")

    st.caption("Аномалии определяются с помощью Isolation Forest, обученного на нормальных режимах.")


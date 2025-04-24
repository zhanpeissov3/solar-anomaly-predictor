# Solar Panel Anomaly Predictor

Это интерактивное приложение на Streamlit, которое предсказывает выработку солнечной энергии по входным данным: температуре воздуха, влажности, скорости ветра, часу суток и установленной мощности.  
Дополнительно приложение определяет, являются ли введённые условия аномальными, что может повлиять на точность прогноза.

## Возможности

- Прогноз генерации солнечной энергии (в кВт·ч)
- Обнаружение нестандартных условий с помощью модели аномалий
- Простая форма ввода и интерпретируемый результат
- Веб-интерфейс без необходимости локальной установки

## Технологии

- Python
- Streamlit
- XGBoost (регрессионная модель)
- Isolation Forest (выявление аномалий)
- pandas, scikit-learn

## Установка локально

```bash
git clone https://github.com/zhanpeissov3/solar-anomaly-predictor.git
cd solar-anomaly-predictor
pip install -r requirements.txt
streamlit run app.py


Онлайн-версия
Приложение доступно по ссылке:
https://solar-panel-anomaly-predictor.streamlit.app/

Состав проекта
app.py                   # Streamlit-приложение
solar_predictor.pkl      # Обученная модель XGBoost
scaler.pkl               # StandardScaler
anomaly_detector.pkl     # Isolation Forest
requirements.txt         # Зависимости проекта


Проект выполнен в рамках Data Science-курса.
Автор: Жанпеисов Кудайберген
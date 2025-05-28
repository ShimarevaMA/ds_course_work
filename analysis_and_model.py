import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_preprocess_data(uploaded_file):
    """Загрузка и предобработка данных"""
    data = pd.read_csv(uploaded_file)
    data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
    data['Type'] = data['Type'].map({'L': 0, 'M': 1, 'H': 2})
    return data


def scale_features(data):
    """Масштабирование числовых признаков"""
    numerical_cols = ['Air temperature [K]', 'Process temperature [K]',
                      'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    return data, scaler, numerical_cols


def train_model(X, y):
    """Обучение модели и оценка результатов"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    return model, X_test, y_test, y_pred, accuracy, auc


def create_prediction_form(model, scaler, numerical_cols):
    """Форма для ввода данных и получения прогноза"""
    with st.form("prediction_form"):
        st.subheader("Параметры оборудования для прогноза")

        col1, col2 = st.columns(2)
        with col1:
            input_type = st.selectbox("Тип оборудования", options=['L', 'M', 'H'])
            air_temp = st.number_input("Температура воздуха [K]", min_value=250.0, max_value=350.0, value=300.0)
            process_temp = st.number_input("Температура процесса [K]", min_value=290.0, max_value=330.0, value=310.0)

        with col2:
            rotation_speed = st.number_input("Скорость вращения [rpm]", min_value=1000.0, max_value=3000.0,
                                             value=1500.0)
            torque = st.number_input("Крутящий момент [Nm]", min_value=10.0, max_value=100.0, value=40.0)
            tool_wear = st.number_input("Износ инструмента [min]", min_value=0, max_value=300, value=100)

        submitted = st.form_submit_button("Прогнозировать состояние")

        if submitted:
            input_data = {
                'Type': input_type,
                'Air temperature [K]': air_temp,
                'Process temperature [K]': process_temp,
                'Rotational speed [rpm]': rotation_speed,
                'Torque [Nm]': torque,
                'Tool wear [min]': tool_wear
            }
            return input_data
    return None


def make_prediction(model, input_data, scaler, numerical_cols):
    """Выполнение прогноза на основе введенных данных"""
    input_df = pd.DataFrame([input_data])
    input_df['Type'] = input_df['Type'].map({'L': 0, 'M': 1, 'H': 2})
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    return prediction, probability


def display_analysis_page():
    """Основная функция для отображения страницы анализа"""
    st.title("🔍 Анализ данных и прогнозная модель")
    st.write("Загрузите данные для обучения модели и прогнозирования отказов оборудования")

    # Загрузка данных
    uploaded_file = st.file_uploader("Выберите CSV файл с данными оборудования", type="csv")

    if uploaded_file:
        data = load_and_preprocess_data(uploaded_file)
        data, scaler, numerical_cols = scale_features(data)

        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']

        model, X_test, y_test, y_pred, accuracy, auc = train_model(X, y)

        # Отображение результатов
        st.success("Модель успешно обучена!")
        st.subheader("Оценка производительности модели")

        col1, col2 = st.columns(2)
        col1.metric("Точность (Accuracy)", f"{accuracy:.2%}")
        col2.metric("ROC-AUC", f"{auc:.2%}")

        st.subheader("Матрица ошибок")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred),
                    annot=True, fmt='d',
                    cmap='coolwarm',
                    ax=ax,
                    cbar=False)
        ax.set_xlabel("Предсказанные классы")
        ax.set_ylabel("Фактические классы")
        st.pyplot(fig)

        # Прогнозирование
        st.divider()
        input_data = create_prediction_form(model, scaler, numerical_cols)

        if input_data:
            prediction, probability = make_prediction(model, input_data, scaler, numerical_cols)

            st.subheader("Результат прогнозирования")
            if prediction == 1:
                st.error(f"❌ Вероятность отказа оборудования: {probability:.2%}")
                st.write("Рекомендуется провести техническое обслуживание")
            else:
                st.success(f"✅ Оборудование работает нормально (Вероятность отказа: {probability:.2%})")
                st.write("Текущее состояние удовлетворительное")
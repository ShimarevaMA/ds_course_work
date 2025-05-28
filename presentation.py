import streamlit as st
import pandas as pd

def create_intro_section():
    st.header("📌 Введение")
    st.markdown("""
    <div padding:20px;border-radius:10px">
        <h4>Цели проекта:</h4>
        <ul>
            <li>Разработка ML-модели для прогнозирования отказов оборудования</li>
            <li>Создание системы раннего предупреждения о возможных сбоях</li>
            <li>Оптимизация затрат на техническое обслуживание</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def create_data_section():
    with st.expander("📊 Описание данных", expanded=False):
        st.markdown("""
        **Источник данных:** AI4I 2020 Predictive Maintenance Dataset (10,000 записей)

        **Ключевые параметры оборудования:**
        - Тип оборудования (L/M/H)
        - Температура воздуха и процесса (K)
        - Скорость вращения (об/мин)
        - Крутящий момент (Нм)
        - Износ инструмента (мин)

        **Статистика отказов:**
        """)

        failure_data = {
            "Тип отказа": ["Износ инструмента", "Теплорассеяние", "Сбой питания"],
            "Количество случаев": [120, 115, 95]
        }
        st.dataframe(pd.DataFrame(failure_data), hide_index=True)


def create_modeling_section():
    with st.expander("⚙️ Моделирование", expanded=False):
        st.markdown("""
        **Архитектура решения:**
        ```mermaid
        graph LR
            A[Сырые данные] --> B{Предобработка};
            B --> C[Обучение модели];
            C --> D[Прогнозирование];
            D --> E[Визуализация];
        ```

        **Используемая модель:**
        ```python
        RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            class_weight="balanced"
        )
        ```
        """)


def create_results_section():
    with st.expander("📈 Результаты", expanded=False):
        st.subheader("Метрики производительности")
        col1, col2 = st.columns(2)
        col1.metric("Точность (Accuracy)", "0.98", delta_color="off")
        col2.metric("ROC-AUC", "0.96", delta_color="off")

        st.subheader("Важность признаков")
        feature_importance = pd.DataFrame({
            "Признак": ["Тип", "Износ", "Крутящий момент", "Температура", "Скорость"],
            "Важность": [0.35, 0.25, 0.20, 0.15, 0.05]
        })
        st.bar_chart(feature_importance.set_index("Признак"))


def create_demo_section():
    with st.expander("🖥️ Демонстрация системы", expanded=False):
        st.markdown("""
        **Основные возможности:**
        - Загрузка и обработка данных в реальном времени
        - Интерактивные прогнозы состояния оборудования
        - Визуализация метрик производительности

        **Пример прогноза:**
        """)

        example_data = {
            "Параметр": ["Тип", "Температура воздуха", "Температура процесса",
                         "Скорость вращения", "Крутящий момент", "Износ инструмента"],
            "Значение": ["L", "300 K", "310 K", "1500 rpm", "40 Nm", "100 min"]
        }
        st.table(pd.DataFrame(example_data))
        st.success("✅ Результат: Оборудование работает нормально (вероятность отказа < 0.01%)")


def create_conclusion_section():
    with st.expander("✅ Заключение", expanded=False):
        st.markdown("""
        **Достигнутые результаты:**
        - Создана модель прогнозирования отказов с точностью 98%
        - Разработано веб-приложение для интерактивного использования модели
        - Определены ключевые факторы, влияющие на отказы оборудования

        **Дальнейшие шаги:**
        - Интеграция с IoT-датчиками оборудования
        - Разработка системы рекомендаций по обслуживанию
        - Адаптация модели для различных типов оборудования
        """)


def display_presentation_page():
    st.title("🎯 Презентация проекта: Прогнозирование отказов оборудования")
    st.divider()

    create_intro_section()
    create_data_section()
    create_modeling_section()
    create_results_section()
    create_demo_section()
    create_conclusion_section()
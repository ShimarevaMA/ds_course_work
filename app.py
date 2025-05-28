import streamlit as st


def initialize_app():
    st.set_page_config(
        page_title="Прогнозирование отказов оборудования",
        layout="wide",
        page_icon="🔧"
    )


def setup_navigation():
    st.sidebar.header("Навигация по разделам")
    page_options = {
        "📈 Анализ и модель": "analysis_and_model",
        "📽️ Презентация": "presentation"
    }
    return st.sidebar.selectbox("Выберите раздел", list(page_options.keys()))


def main():
    initialize_app()
    selected_page = setup_navigation()

    if selected_page == "📈 Анализ и модель":
        from analysis_and_model import display_analysis_page
        display_analysis_page()
    else:
        from presentation import display_presentation_page
        display_presentation_page()


if __name__ == "__main__":
    main()
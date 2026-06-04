import streamlit as st

st.set_page_config(
    page_title="KC House Price Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS ="""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F2C9DC 0%, #E8A0BF 100%);
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] a {
    color: #5C2740 !important;
}
.main .block-container { padding-top: 1.5rem; }
h1 {
    color: #7A2453 !important;
    font-weight: 800;
    border-bottom: 3px solid #E8789A;
    padding-bottom: .4rem;
    margin-bottom: 1rem;
}
h2 { color: #9B3D6A !important; font-weight: 700; }
h3 { color: #B85A85 !important; }
[data-testid="metric-container"] {
    background: #FDF0F5;
    border: 1px solid #ECC5D8;
    border-radius: 10px;
    padding: .6rem 1rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.title("KC House Price Predictor")
st.markdown("**Дисциплина:** Машинное обучение и большие данные")
st.markdown(
"**Тема РГР:** Разработка Web-приложения (дашборда)"
"для инференса моделей ML и анализа данных"
)
st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("#### Разработчик")
    st.info("Информация об авторе и теме РГР")
with c2:
    st.markdown("#### Датасет")
    st.info("Описание данных King County, EDA")
with c3:
    st.markdown("#### Визуализации")
    st.info("Графики зависимостей и распределений")
with c4:
    st.markdown("#### Инференс")
    st.info("Предсказание стоимости дома от 6 моделей ML")

st.markdown("---")
st.caption("Омский государственный технический университет · 2026")

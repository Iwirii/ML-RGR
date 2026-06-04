import streamlit as st
import os

st.set_page_config(page_title="Разработчик", page_icon="", layout="wide")

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
.dev-card {
    background: #FDF0F5;
    border-left: 5px solid #C96B96;
    border-radius: 8px;
    padding: 1rem 1.4rem;
    margin-bottom: .8rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.title("О разработчике")

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    for path in ["assets/photo.png","assets/photo.jpg"]:
        if os.path.exists(path):
            st.image(path, width=260)
            break
    else:
        st.markdown(
"<div style='width:260px;height:260px;background:#D0DCF0;"
"border-radius:12px;display:flex;align-items:center;"
"justify-content:center;font-size:5rem;'></div>",
            unsafe_allow_html=True,
)

with col2:
    st.markdown("## Фирстова Ксения Максимовна")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Группа","ФИТ-241")
    c2.metric("Год","2026")
    c3.metric("Моделей","6")
    c4.metric("Датасет","KC Houses")

    st.markdown("---")
    st.markdown("### Тема РГР")
    st.markdown(
"<div class='dev-card'>"
"Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML"
"и анализа данных на основе датасета цен на недвижимость"
" округа Кинг (King County, Seattle)."
"</div>",
        unsafe_allow_html=True,
)

    st.markdown("### Университет")
    st.markdown(
"<div class='dev-card'>"
"Омский государственный технический университет ·"
"Дисциплина: Машинное обучение и большие данные"
"</div>",
        unsafe_allow_html=True,
)

st.markdown("---")
st.markdown("### О проекте")

r1, r2, r3 = st.columns(3)
with r1:
    st.markdown("#### Данные")
    st.markdown(
"Датасет **kc_house_data** содержит 21 613 объявлений"
"о продаже домов в округе Кинг (штат Вашингтон, США)"
"за 2014–2015 годы."
)
with r2:
    st.markdown("#### Модели")
    st.markdown(
"Обучено **6 моделей ML**: полиномиальная регрессия,"
"градиентный бустинг, CatBoost, бэггинг, стэкинг"
"и полносвязная нейронная сеть (Keras)."
)
with r3:
    st.markdown("#### Стек")
    st.markdown(
"**Python · Streamlit · Scikit-learn ·"
"CatBoost · TensorFlow/Keras ·"
"Pandas · Matplotlib · Seaborn · Optuna**"
)

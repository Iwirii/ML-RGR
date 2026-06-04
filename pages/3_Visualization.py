import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Визуализации", page_icon="", layout="wide")

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
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

ACCENT   ="#1A3460"
ACCENT2  ="#E06B2F"
C1, C2   ="#3A6DAF","#E06B2F"

st.title("Визуализации зависимостей")

@st.cache_data
def load_data():
    path ="data/kc_house_data_cleaned.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df = load_data()
if df is None:
    st.warning("Файл `data/kc_house_data_cleaned.csv` не найден.")
    st.stop()

TARGET   ="price"
num_cols = df.select_dtypes(include=np.number).columns.tolist()

# График 1: Гистограмма цен
st.markdown("## 1. Распределение цен на недвижимость")
st.markdown("Гистограмма показывает, что большинство домов стоят до $600 000.")
st.markdown("Распределение правосторонне скошено — присутствуют единичные объекты стоимостью более $5 млн.")

col1, col2 = st.columns([3, 1])
with col2:
    bins = st.slider("Столбцов", 10, 100, 50, key="hist_bins")
    log_scale = st.checkbox("Лог. шкала X", value=False)
with col1:
    fig, ax = plt.subplots(figsize=(8, 4))
    data_plot = np.log1p(df[TARGET]) if log_scale else df[TARGET]
    xlabel ="log(price + 1)"if log_scale else"Цена (USD)"
    ax.hist(data_plot.dropna(), bins=bins, color=ACCENT, edgecolor="white", linewidth=0.4)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Количество домов", fontsize=12)
    ax.set_title("Распределение цен на дома (King County)", fontsize=13, color=ACCENT)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# График 2: Корреляционная матрица
st.markdown("## 2. Корреляционная матрица ключевых признаков")
st.markdown(
"Тепловая карта показывает попарные корреляции Пирсона."
"`sqft_living` и `grade` имеют наибольшую положительную корреляцию с ценой."
)

corr_cols = ["price","sqft_living","sqft_lot","bedrooms","bathrooms",
"floors","grade","condition","age","sqft_above"]
corr_cols = [c for c in corr_cols if c in df.columns]
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(
    df[corr_cols].corr(), annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, linewidths=0.4, ax=ax,
    annot_kws={"size": 8},
)
ax.set_title("Корреляционная матрица", fontsize=13, color=ACCENT)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")

# График 3: Scatter — признак vs цена
st.markdown("## 3. Зависимость признака от цены")
st.markdown("Диаграмма рассеяния позволяет оценить линейность и разброс.")

col1, col2 = st.columns([3, 1])
with col2:
    available = [c for c in num_cols if c != TARGET]
    default_x ="sqft_living"if"sqft_living"in available else available[0]
    feature_x = st.selectbox("Признак (ось X)", available, index=available.index(default_x))
    sample_n  = st.slider("Размер выборки", 500, min(5000, len(df)), 3000, step=500)
    alpha_val = st.slider("Прозрачность", 0.05, 1.0, 0.25, step=0.05)
with col1:
    sample = df[[feature_x, TARGET]].dropna().sample(min(sample_n, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(sample[feature_x], sample[TARGET],
               alpha=alpha_val, color=ACCENT, s=12, edgecolors="none")
    ax.set_xlabel(feature_x, fontsize=12)
    ax.set_ylabel("Цена (USD)", fontsize=12)
    ax.set_title(f"{feature_x}  vs  price", fontsize=13, color=ACCENT)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# График 4: Boxplot цены по числу спален
st.markdown("## 4. Распределение цен по числу спален")
st.markdown(
"Ящик с усами показывает, как цена растёт с числом спален."
"Дома с 7+ спальнями имеют широкий разброс из-за малой выборки."
)

col1, col2 = st.columns([3, 1])
with col2:
    price_cap = st.number_input("Макс. цена (USD)", value=3_000_000, step=200_000)
with col1:
    df_box = df[df[TARGET] <= price_cap].copy()
    df_box["bedrooms_clipped"] = df_box["bedrooms"].clip(upper=8)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(
        data=df_box, x="bedrooms_clipped", y=TARGET,
        palette="Blues", ax=ax, flierprops={"markersize": 2},
)
    ax.set_xlabel("Число спален (8+ объединены)", fontsize=12)
    ax.set_ylabel("Цена (USD)", fontsize=12)
    ax.set_title("Распределение цен по числу спален", fontsize=13, color=ACCENT)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# График 5: Средняя цена по году постройки
st.markdown("## 5. Средняя цена по году постройки")
st.markdown(
"Линейный график показывает динамику цен в зависимости от года постройки дома."
"Дома, построенные в 1900–1940-х, нередко дороже из-за расположения в историческом центре Сиэтла."
)

if"yr_built"in df.columns:
    yr_price = (
        df.groupby("yr_built")[TARGET]
        .agg(["mean","median","count"])
        .reset_index()
        .query("count >= 5")
)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(yr_price["yr_built"], yr_price["mean"],
            color=C2, linewidth=2, label="Средняя цена")
    ax.plot(yr_price["yr_built"], yr_price["median"],
            color=C1, linewidth=2, linestyle="--", label="Медианная цена")
    ax.fill_between(yr_price["yr_built"],
                    yr_price["mean"], yr_price["median"],
                    alpha=0.12, color=ACCENT)
    ax.set_xlabel("Год постройки", fontsize=12)
    ax.set_ylabel("Цена (USD)", fontsize=12)
    ax.set_title("Динамика цен по году постройки", fontsize=13, color=ACCENT)
    ax.legend(fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
else:
    st.info("Признак `yr_built` не найден в датасете.")

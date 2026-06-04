import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Датасет", page_icon="", layout="wide")

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

st.title("Описание датасета")

@st.cache_data
def load_data():
    path ="data/kc_house_data_cleaned.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df = load_data()

st.markdown("## Предметная область")
st.markdown("""
Датасет **King County House Sales** содержит информацию о **21 613** сделках купли-продажи
жилой недвижимости в округе Кинг (штат Вашингтон, США) в период с мая 2014 по май 2015 года.

**Целевая переменная:** `price` — стоимость дома в долларах США (USD).

**Задача:** регрессия — предсказание рыночной цены дома по его техническим характеристикам и местоположению.
""")

st.markdown("---")
st.markdown("## Описание признаков")

features_info = pd.DataFrame({
"Признак": [
"bedrooms","bathrooms","sqft_living","sqft_lot",
"floors","waterfront","view","condition","grade",
"sqft_above","sqft_basement","yr_built","zipcode",
"lat","long","sqft_living15","sqft_lot15",
"day / month / year","age","was_renovated",
"years_since_renovation","price"
],
"Тип": [
"целое","дробное","целое","целое",
"дробное","бинарный","целое","целое","целое",
"целое","целое","целое","категориальный",
"дробное","дробное","целое","целое",
"целое","целое","бинарный",
"целое","числовой"
],
"Описание": [
"Количество спален","Количество ванных комнат","Жилая площадь (кв. фут)",
"Площадь участка (кв. фут)","Количество этажей","Выход к воде (0/1)",
"Качество вида (0 — нет, 4 — отличный)","Техническое состояние дома",
"Оценка качества строительства и отделки","Площадь надземной части (кв. фут)",
"Площадь подвала (кв. фут)","Год постройки","Почтовый индекс района",
"Географическая широта","Географическая долгота",
"Средняя жилая площадь 15 ближайших домов","Средняя площадь участков 15 соседей",
"Дата объявления о продаже","Возраст дома (лет) на момент продажи",
"Был ли дом отремонтирован (0/1)","Лет с последнего ремонта",
"Цена продажи (USD) — целевая переменная"
]
})
st.dataframe(features_info, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("## Инженерия признаков и предобработка")
st.markdown("""
В ходе подготовки данных выполнены следующие шаги:

1. **Производные признаки** — добавлено 5 новых переменных на основе существующих:
   - `sqft_ratio = sqft_living / sqft_lot` — доля жилой площади от площади участка
   - `total_rooms = bedrooms + bathrooms` — общее число комнат
   - `rooms_per_floor = bedrooms / floors` — спален на этаж
   - `renovated_age = age − years_since_renovation` — возраст дома с момента последнего ремонта
   - `living_vs_15 = sqft_living / sqft_living15` — отношение к средней площади соседей
2. **Масштабирование** — `StandardScaler` применён для полиномиальной регрессии и нейросети.
3. **Разбивка** — 80% обучение / 20% тест, `random_state=42`.
""")

st.markdown("---")
st.markdown("## Результаты разведочного анализа (EDA)")

st.markdown("### Общая характеристика данных")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Строк", f"{df.shape[0]:,}")
c2.metric("Признаков", df.shape[1] - 1)
c3.metric("Пропусков", int(df.isnull().sum().sum()))
c4.metric("Целевая переменная","price")

st.markdown("### Анализ целевой переменной (price)")
col1, col2, col3 = st.columns(3)
col1.metric("Средняя цена", f"${df['price'].mean():,.0f}")
col2.metric("Медианная цена", f"${df['price'].median():,.0f}")
col3.metric("Максимальная цена", f"${df['price'].max():,.0f}")
st.markdown("""
**Вывод:** Распределение цен имеет положительную асимметрию (среднее > медианы).
Большинство домов находятся в диапазоне 300 000 - 600 000 долларов, но присутствуют дорогие объекты,
которые смещают среднее значение вверх.
""")

st.markdown("### Ключевые факторы, влияющие на цену")
st.markdown("""
На основе корреляционного анализа признаков с целевой переменной `price` были выявлены следующие зависимости:

| Степень влияния | Признак | Описание |
|----------------|---------|----------|
| Сильное положительное | `sqft_living` | Жилая площадь (корреляция ~0.70) |
| Сильное положительное | `grade` | Оценка качества дома (корреляция ~0.67) |
| Умеренное положительное | `bathrooms` | Количество ванных комнат (корреляция ~0.53) |
| Слабое положительное | `view` | Качество вида из окна (корреляция ~0.40) |
| Слабое положительное | `waterfront` | Наличие вида на воду (корреляция ~0.27) |
| Очень слабое | `age` | Возраст дома (корреляция < 0.1) |
| Очень слабое | `sqft_lot` | Площадь участка (корреляция < 0.1) |
""")

st.markdown("### Обнаруженные выбросы")
st.markdown("""
- **price и sqft_living:** Присутствуют выбросы в виде очень дорогих и больших домов (дороже 3 млн долларов или площадью более 5000 кв. футов)
- **bedrooms:** Обнаружен дом с 33 спальнями, что является явной аномалией. Данный выброс был исключён из обучающей выборки для предотвращения искажения модели
- **waterfront и view:** Редкие значения (наличие воды, высокий балл вида) являются не выбросами, а ценными премиальными характеристиками
""")

st.markdown("---")
st.markdown("## Просмотр данных")

if df is not None:
    st.markdown("### Первые строки датасета")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Статистика числовых признаков")
    st.dataframe(df.describe().round(2), use_container_width=True)
else:
    st.warning("Файл `data/kc_house_data_cleaned.csv` не найден.")
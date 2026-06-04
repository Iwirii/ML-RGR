import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Инференс", page_icon="", layout="wide")

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
.result-box {
    background: linear-gradient(135deg, #C96B96 0%, #E8A0BF 100%);
    border-radius: 14px;
    padding: 1.8rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-price {
    font-size: 2.6rem;
    font-weight: 800;
    color: #5C2740;
    margin: 0;
    line-height: 1.2;
}
.result-label {
    color: #F5D0E5;
    font-size: 1rem;
    margin-bottom: .4rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Районы King County (zipcode / lat / long) 
NEIGHBORHOODS = {
"Seattle — Capitol Hill (98122)":    (47.6141, -122.3141, 98122),
"Seattle — Ballard (98117)":         (47.6815, -122.3845, 98117),
"Seattle — Beacon Hill (98108)":     (47.5680, -122.2960, 98108),
"Bellevue (98004)":                  (47.6101, -122.2015, 98004),
"Kirkland (98033)":                  (47.6815, -122.2087, 98033),
"Redmond (98052)":                   (47.6740, -122.1215, 98052),
"Renton (98055)":                    (47.4799, -122.2171, 98055),
"Mercer Island (98040)":             (47.5707, -122.2221, 98040),
"Issaquah (98027)":                  (47.5301, -122.0329, 98027),
"Kent (98031)":                      (47.3809, -122.2348, 98031),
}

# Загрузка моделей 
@st.cache_resource
def load_models():
    models, errors = {}, []

    pkl_models = {
        "Полиномиальная регрессия": "models/polynomial.pkl",
        "Gradient Boosting":        "models/gradient_boosting.pkl",
        "BaggingRegressor":         "models/bagging.pkl",
        "StackingRegressor":        "models/stacking.pkl",
    }
    for name, path in pkl_models.items():
        if os.path.exists(path):
            with open(path,"rb") as f:
                models[name] = ("pkl", pickle.load(f))
        else:
            errors.append(f"Не найден: `{path}`")

    cb_path ="models/catboost.cbm"
    if os.path.exists(cb_path):
        try:
            from catboost import CatBoostRegressor
            cb = CatBoostRegressor()
            cb.load_model(cb_path)
            models["CatBoost"] = ("catboost", cb)
        except Exception as e:
            errors.append(f"CatBoost: {e}")
    else:
        errors.append("Не найден: `models/catboost.cbm`")

    for keras_name in ("models/keras_nn.keras","models/keras_rmsprop.keras"):
        if os.path.exists(keras_name):
            try:
                import tensorflow as tf
                tf.get_logger().setLevel("ERROR")
                models["Нейросеть Keras"] = ("keras", tf.keras.models.load_model(keras_name))
            except Exception as e:
                errors.append(f"Keras: {e}")
            break
    else:
        errors.append("Не найден: keras_nn.keras / keras_rmsprop.keras")

    return models, errors


@st.cache_resource
def load_scaler():
    path ="models/scaler.pkl"
    if not os.path.exists(path):
        return None
    with open(path,"rb") as f:
        return pickle.load(f)


def add_feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["sqft_ratio"]      = X["sqft_living"] / X["sqft_lot"].clip(lower=1)
    X["total_rooms"]     = X["bedrooms"] + X["bathrooms"]
    X["rooms_per_floor"] = X["bedrooms"] / X["floors"].clip(lower=1)
    X["renovated_age"]   = X["age"] - X["years_since_renovation"]
    X["living_vs_15"]    = X["sqft_living"] / X["sqft_living15"].clip(lower=1)
    return X


def make_prediction(mtype, mobj, X_raw: pd.DataFrame, scaler) -> np.ndarray:
    if mtype =="keras":
        X_sc = pd.DataFrame(scaler.transform(X_raw), columns=X_raw.columns)
        return np.expm1(mobj.predict(X_sc, verbose=0).flatten())
    return mobj.predict(X_raw)


def fmt(usd: float) -> str:
    return f"${usd:,.0f}"


# Загрузка 
models, load_errors = load_models()
scaler = load_scaler()

if load_errors:
    with st.expander("Предупреждения при загрузке моделей", expanded=False):
        for e in load_errors:
            st.warning(e)

if not models:
    st.error("Ни одна модель не загружена. Убедитесь, что папка `models/` содержит обученные модели.")
    st.stop()

# Заголовок 
st.title("Предсказание стоимости дома")

# Шаг 1: выбор модели 
st.markdown("## Шаг 1 — Выберите модель")
model_name = st.selectbox("Модель ML", list(models.keys()))
model_type, model_obj = models[model_name]

st.markdown("---")

# Шаг 2: ввод данных 
st.markdown("## Шаг 2 — Введите данные")
input_method = st.radio(
"Способ ввода",
    ["Ввести вручную","Загрузить CSV-файл"],
    horizontal=True,
)

# Ручной ввод 
if input_method =="Ввести вручную":
    st.markdown("### Характеристики объекта недвижимости")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Площадь**")
        sqft_living  = st.number_input("Жилая площадь (кв. фут)", 300, 13_500, 2_000, step=50)
        sqft_lot     = st.number_input("Площадь участка (кв. фут)", 500, 1_700_000, 7_500, step=100)
        sqft_above   = st.number_input("Надземная площадь (кв. фут)", 300, 9_410,
                                        min(sqft_living, 9_410), step=50)
        sqft_basement = sqft_living - sqft_above
        st.caption(f"Площадь подвала (авто): **{sqft_basement} кв. фут**")
        sqft_living15 = st.number_input("Жил. площадь соседей (кв. фут)", 399, 6_210, 2_000, step=50)
        sqft_lot15    = st.number_input("Площадь участка соседей (кв. фут)", 651, 871_200, 7_500, step=100)

    with col2:
        st.markdown("**Комнаты и качество**")
        bedrooms  = st.slider("Спален", 0, 10, 3)
        bathrooms = st.slider("Ванных", 0.0, 8.0, 2.0, step=0.25)
        floors    = st.selectbox("Этажей", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        grade     = st.slider("Качество стройки (grade, 1–13)", 1, 13, 7)
        condition = st.slider("Состояние (1–5)", 1, 5, 3)
        view      = st.slider("Вид из окон (0–4)", 0, 4, 0)
        waterfront = int(st.checkbox("Выход к воде"))

    with col3:
        st.markdown("**Возраст и расположение**")
        yr_built      = st.number_input("Год постройки", 1900, 2015, 1990)
        was_renovated = st.checkbox("Дом был отремонтирован")
        yr_renovated  = 0
        if was_renovated:
            yr_renovated = st.number_input("Год ремонта", 1950, 2015, 2000)

        neighborhood = st.selectbox("Район King County", list(NEIGHBORHOODS.keys()))
        lat, lon, zipcode = NEIGHBORHOODS[neighborhood]
        st.caption(f"lat {lat:.4f} · long {lon:.4f} · zip {zipcode}")

    st.markdown("---")
    st.markdown("### Шаг 3 — Получить предсказание")

    if st.button("Рассчитать стоимость дома", type="primary", use_container_width=True):
        listing_year = 2015
        age = listing_year - yr_built
        years_since_reno = (listing_year - yr_renovated) if was_renovated else 0

        raw = {
"bedrooms": bedrooms,"bathrooms": bathrooms,
"sqft_living": sqft_living,"sqft_lot": sqft_lot,
"floors": floors,"waterfront": waterfront,
"view": view,"condition": condition,"grade": grade,
"sqft_above": sqft_above,"sqft_basement": sqft_basement,
"yr_built": yr_built,"zipcode": zipcode,
"lat": lat,"long": lon,
"sqft_living15": sqft_living15,"sqft_lot15": sqft_lot15,
"day": 2,"month": 5,"year": listing_year,
"age": age,
"was_renovated": int(was_renovated),
"years_since_renovation": years_since_reno,
        }
        X_raw = add_feature_engineering(pd.DataFrame([raw]))

        try:
            pred = float(make_prediction(model_type, model_obj, X_raw, scaler)[0])
            pred = max(0.0, pred)

            st.markdown(
                f"<div class='result-box'>"
                f"<p class='result-label'>Прогнозируемая стоимость дома</p>"
                f"<p class='result-price'>{fmt(pred)} USD</p>"
                f"</div>",
                unsafe_allow_html=True,
)

            st.markdown(f"**Модель:** {model_name}")
            m2, m3, m4 = st.columns(3)
            m2.metric("Возраст дома", f"{age} лет")
            m3.metric("Жилая площадь", f"{sqft_living:,} кв. фут")
            m4.metric("Grade", grade)

        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")

# CSV
else:
    st.markdown("### Загрузка CSV-файла")
    st.info(
        "Можно загрузить CSV как с колонкой `price`, так и без неё. "
        "Если `price` присутствует — в результате появится колонка с реальной стоимостью для сравнения."
    )

    uploaded = st.file_uploader("Выберите CSV-файл", type=["csv"])

    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
            has_target = "price" in df_upload.columns
            st.markdown(
                f"**Строк:** {len(df_upload)} · "
                f"**Признаков:** {df_upload.shape[1]} · "
                f"**Реальные цены:** {'да' if has_target else 'нет'}"
            )
            st.dataframe(df_upload.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Ошибка чтения файла: {e}")
            st.stop()

        st.markdown("---")
        if st.button("Рассчитать стоимость для всех строк", type="primary", use_container_width=True):
            try:
                # отделяем целевую переменную, если есть
                real_prices = None
                if has_target:
                    real_prices = df_upload["price"].values
                    X_raw = add_feature_engineering(df_upload.drop(columns=["price"]))
                else:
                    X_raw = add_feature_engineering(df_upload.copy())

                preds = make_prediction(model_type, model_obj, X_raw, scaler)
                preds = np.maximum(preds, 0)

                df_result = df_upload.copy()
                df_result["predicted_price"] = preds.round(0).astype(int)

                if real_prices is not None:
                    df_result["actual_price"] = real_prices
                    df_result["error"] = (df_result["predicted_price"] - df_result["actual_price"]).abs()
                    df_result["error_%"] = (
                        df_result["error"] / df_result["actual_price"].replace(0, np.nan) * 100
                    ).round(1)
                    # переставляем колонки: сначала predicted, потом actual
                    cols = [c for c in df_result.columns
                            if c not in ("predicted_price", "actual_price", "error", "error_%", "price")]
                    df_result = df_result[["predicted_price", "actual_price", "error", "error_%"] + cols]

                st.success(f"Предсказания получены для {len(preds)} строк")
                st.dataframe(df_result, use_container_width=True)

                if real_prices is not None:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Средняя предсказанная", fmt(preds.mean()) + " USD")
                    c2.metric("Средняя реальная", fmt(real_prices.mean()) + " USD")
                    c3.metric("Средняя ошибка", fmt(df_result["error"].mean()) + " USD")
                    c4.metric("Средняя ошибка %", f"{df_result['error_%'].mean():.1f}%")
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Средняя цена", fmt(preds.mean()) + " USD")
                    c2.metric("Мин. цена",    fmt(preds.min())  + " USD")
                    c3.metric("Макс. цена",   fmt(preds.max())  + " USD")

                st.download_button(
                    "Скачать результаты CSV",
                    df_result.to_csv(index=False).encode("utf-8"),
                    "predictions.csv", "text/csv",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Ошибка при предсказании: {e}")

# Dashboard Final equipo — Proyecto Airbnb (By Raymundo Díaz + IA + Profe Freddy)
# Versión final optimizada y revisada

##########
# Importar librerías
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    confusion_matrix, accuracy_score, precision_score,
    recall_score, roc_auc_score, roc_curve, classification_report, f1_score,
    precision_recall_curve, average_precision_score, balanced_accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

##########
# Configuración global
st.set_page_config(
    page_title="Airbnb (Data Web)",
    page_icon="assets/icon.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta Airbnb
AIRBNB_RED   = "#FF5A5F"
AIRBNB_TEAL  = "#00A699"
AIRBNB_ORANGE= "#FC642D"
AIRBNB_GRAY  = "#BFBFBF"
AIRBNB_DARK_BG = "#0E1117"
AIRBNB_CARD   = "#151A22"
AIRBNB_BORDER = "#232A35"
CONT_GRADIENT = "Reds"

##########
# CSS Look & Feel Airbnb
st.markdown(f"""
<style>
.block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; }}

/* Fondo degradado unificado */
html, body, [data-testid="stAppViewContainer"], section[data-testid="stSidebar"] {{
    background: radial-gradient(circle at 30% 30%, #131722 0%, #0E1117 100%) !important;
    color: white !important;
}}
section[data-testid="stSidebar"] {{
    border-right: 1px solid {AIRBNB_BORDER};
}}

/* Tarjetas KPI */
.air-card {{
    border: 1px solid {AIRBNB_BORDER};
    border-radius:16px; padding:1rem;
    background:{AIRBNB_CARD};
}}

/* Botones */
.stButton>button {{
    background:{AIRBNB_RED}; color:white; border-radius:12px; border:none;
    padding:.6rem 1rem; font-weight:600;
}}
.stButton>button:hover {{ opacity:.9 }}

/* Tablas */
.stDataFrame, .stTable {{ color: white !important; }}
</style>
""", unsafe_allow_html=True)

##########
# Plotly: plantilla Airbnb
AIRBNB_COLORWAY = ["#FF5A5F", "#00A699", "#FC642D", "#BFBFBF", "#767676"]
pio.templates["airbnb_dark"] = pio.templates["plotly_dark"]
pio.templates["airbnb_dark"].layout.colorway = AIRBNB_COLORWAY
px.defaults.template = "airbnb_dark"
px.defaults.color_continuous_scale = CONT_GRADIENT
px.defaults.height = 420

##########
# Multi-país
COUNTRY_FILES = {
    "Alemania": "Berlin_Final.csv",
    "Valencia": "Valencia_Final.csv",
    "Estocolmo": "Estocolmo_Final.csv",
    "Mexico": "Mexico_Final.csv",
}

COUNTRY_IMAGES = {
    "Alemania": ["assets/Berlin1.jpg", "assets/Berlin3.jpg", "assets/Berlin2.jpg"],
    "Valencia": ["assets/Valencia1.jpg", "assets/Valencia2.jpg", "assets/Valencia3.jpg"],
    "Estocolmo": ["assets/Estocolmo1.jpg", "assets/Estocolmo2.jpg", "assets/Estocolmo3.jpg"],
    "Mexico": ["assets/Mexico1.jpg", "assets/Mexico2.jpg", "assets/Mexico3.jpg"],
}

##########
# Normalización
BIN_TRUE = {"t","true","True",1,"1",True}
BIN_FALSE= {"f","false","False",0,"0",False}

def _normalize_binary(series):
    s = series.copy()
    return s.apply(lambda v: 1 if v in BIN_TRUE else (0 if v in BIN_FALSE else np.nan)).astype("float")

def _normalize_df(df_raw):
    df = df_raw.copy()

    # 1) Normaliza nombres de columnas (quitan espacios accidentales)
    df.columns = df.columns.str.strip()

    # 2) Drop de *cualquier* columna Unnamed
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)]

    # 3) Drops opcionales que ya tenías
    df = df.drop(['latitude','longitude','first_review','last_review','host_since', 'price', 'estimated_revenue_l365d'],
                 axis=1, errors="ignore")

    # 4) Tipos
    if 'host_id' in df.columns:
        df['host_id'] = df['host_id'].astype(str)

    # 5) Normaliza binarias
    for col in ['host_is_superhost','host_identity_verified','instant_bookable']:
        if col in df.columns:
            df[col] = _normalize_binary(df[col])

    # 6) A numérico (coerce => NaN si hay “90%”, etc.)
    for col in ['host_response_rate','host_acceptance_rate','price','estimated_revenue_l365d']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def _clean_xy(df_base, y_col, x_cols):
    """
    Devuelve X, y sin NaN/Inf y un conteo de filas filtradas.
    """
    work = df_base[x_cols + [y_col]].replace([np.inf, -np.inf], np.nan)
    before = len(work)
    work = work.dropna()
    after = len(work)
    X = work[x_cols].to_numpy(dtype=float)
    y = work[y_col].to_numpy(dtype=float)
    return X, y, before - after

@st.cache_data(show_spinner=False)
def load_country_df(country: str):
    path = COUNTRY_FILES[country]
    raw = pd.read_csv(path)
    df = _normalize_df(raw)
    Lista = [
        'host_is_superhost','host_identity_verified','host_response_time',
        'host_response_rate','host_acceptance_rate','host_total_listings_count',
        'host_verifications','room_type','property_type','price_cat'
    ]
    return df, Lista

# Carga inicial
df, Lista = load_country_df("Alemania")

##########
# Header
col_logo, col_title = st.columns([1,5], vertical_alignment="center")
with col_logo:
    st.image("assets/Logo3.jpg", width=90)
with col_title:
    st.markdown("""
        # Airbnb Data Analysis
        <span style="color:#767676">Listados, precios y comportamiento de oferta</span>
    """, unsafe_allow_html=True)

##########
# Sidebar
st.sidebar.image("assets/Logoo.jpg", use_container_width=True)
st.sidebar.caption("Análisis exploratorio y modelos")
st.sidebar.markdown("---")
modo_presentacion = st.sidebar.toggle("Modo presentación", value=False)
country = st.sidebar.selectbox("País", list(COUNTRY_FILES.keys()), index=0)
df, Lista = load_country_df(country)
View = st.sidebar.selectbox(
    label='Tipo de análisis',
    options=['Extracción de Características', 'Regresión Lineal', 'Regresión No Lineal', 'Regresión Logística', 'Comparar países'],
    index=0
)

##########################################################################################
# Vista 1 — Extracción de características
if View == "Extracción de Características":
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="air-card">', unsafe_allow_html=True)
        st.metric("Filas", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="air-card">', unsafe_allow_html=True)
        st.metric("Tipos de propiedad", df['property_type'].nunique() if 'property_type' in df.columns else "—")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="air-card">', unsafe_allow_html=True)
        med_price = np.nanmedian(df['price_eur']) if 'price_eur' in df.columns else np.nan
        st.metric("Mediana de precio (€)", f"€{med_price:,.0f}" if np.isfinite(med_price) else "—")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="air-card">', unsafe_allow_html=True)
        superhosts = int((df['host_is_superhost'] == 1).sum()) if 'host_is_superhost' in df.columns else 0
        st.metric("Superhosts", superhosts)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    Variable_Cat = st.sidebar.selectbox("Variable categórica a analizar", options=Lista)
    Tabla_frecuencias = df[Variable_Cat].value_counts(dropna=False).reset_index().head(10)
    Tabla_frecuencias.columns = ['categorias', 'frecuencia']

    st.title("Extracción de Características")
    st.caption('Se muestran máximo las 10 categorías con más frecuencia.')

    Contenedor_A, Contenedor_B = st.columns(2)
    with Contenedor_A:
        st.subheader("Distribución por categoría (Bar Plot)")
        fig_bar = px.bar(Tabla_frecuencias, x='categorias', y='frecuencia', color='categorias')
        st.plotly_chart(fig_bar, use_container_width=True)
    with Contenedor_B:
        st.subheader("Proporción por categoría (Pie Chart)")
        fig_pie = px.pie(Tabla_frecuencias, names='categorias', values='frecuencia')
        st.plotly_chart(fig_pie, use_container_width=True)

    Contenedor_C, Contenedor_D = st.columns(2)
    with Contenedor_C:
        st.subheader("Gráfico tipo anillo")
        fig_donut = px.pie(Tabla_frecuencias, names='categorias', values='frecuencia', hole=0.5)
        st.plotly_chart(fig_donut, use_container_width=True)
    with Contenedor_D:
        st.subheader("Tendencia acumulada (Área)")
        fig_area = px.area(Tabla_frecuencias.sort_values(by='frecuencia', ascending=False),
                           x='categorias', y='frecuencia')
        st.plotly_chart(fig_area, use_container_width=True)

    st.markdown("---")
    st.subheader("Análisis más profundo")

    if Variable_Cat in ['room_type', 'property_type', 'price_cat'] and 'price' in df.columns:
        st.write("**Relación entre categorías y precio (Boxplot):**")
        fig_box = px.box(df, x=Variable_Cat, y='price', color=Variable_Cat)
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.write("**Heatmap de proporciones:**")
        heat_df = pd.crosstab(index=df[Variable_Cat], columns='count', normalize='columns') * 100
        fig_heat = px.imshow(heat_df, color_continuous_scale=CONT_GRADIENT, title="Proporción por categoría")
        st.plotly_chart(fig_heat, use_container_width=True)

    if not modo_presentacion:
        st.markdown("---")
        st.subheader("Tabla de frecuencias")
        st.dataframe(Tabla_frecuencias.style.background_gradient(cmap='Reds'), use_container_width=True)

    st.markdown(f"**Galería:** {country} — Airbnb")
    imgs = COUNTRY_IMAGES.get(country, [])
    gcols = st.columns(3)
    for i, path in enumerate(imgs[:3]):
        with gcols[i]:
            try:
                st.image(path, use_container_width=True)
            except Exception:
                st.write("🖼️ Imagen no encontrada")



##########################################################################################
# Vista 2 
if View == "Regresión Lineal":
    st.title("Regresión Lineal")

    # Variables numéricas disponibles
    numeric_df = df.select_dtypes(include=['float', 'float64', 'int', 'int64']).copy()
    Lista_num = list(numeric_df.columns)

    # Lineal simple
    st.subheader("Regresión lineal simple")
    colL, colR = st.columns(2)
    with colL:
        Variable_y = st.selectbox("Variable dependiente (Y)", options=Lista_num, key="rl_y")
    with colR:
        Variable_x = st.selectbox("Variable independiente (X)", options=Lista_num, key="rl_x")

    # Ajuste
    X, y, dropped = _clean_xy(numeric_df, Variable_y, [Variable_x])
    if dropped > 0 and not modo_presentacion:
        st.info(f"Se descartaron {dropped} filas con NaN/Inf para el ajuste.")

    if len(y) < 3:
        st.error("No hay suficientes filas válidas para ajustar el modelo.")
        st.stop()

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)


    # Métricas
    r2 = r2_score(y, y_pred)
    coef_Deter_simple = model.score(X= X, y= y)
    coef_Correl_simple = np.sqrt(abs(coef_Deter_simple))

    # Coeficientes
    coef_df_simple = pd.DataFrame({
        "Variable": [Variable_x],
        "Coeficiente": [model.coef_[0]],
        "Intercepto": [model.intercept_],
        "R": [coef_Correl_simple],
        "R^2": [coef_Deter_simple]
    })

    if not modo_presentacion:
        st.dataframe(coef_df_simple, use_container_width=True)

    # Gráfica: dispersión + recta y_pred
    fig_scat = px.scatter(numeric_df, x=Variable_x, y=Variable_y, opacity=0.6, title="Dispersión y recta ajustada")
    # Línea predicha ordenando por X
    order_idx = np.argsort(X[:, 0])
    fig_scat.add_trace(go.Scatter(
        x=X[order_idx, 0], y=y_pred[order_idx],
        mode="lines", name="Predicción de Y"
    ))
    st.plotly_chart(fig_scat, use_container_width=True)

    # Residuales
    resid = y - y_pred
    fig_res = px.scatter(x=y_pred, y=resid, labels={"x":"Ŷ", "y":"Residual"},
                         title="Residuos vs Predicción (diagnóstico)")
    fig_res.add_hline(y=0, line_dash="dot")
    st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("---")

    # Lineal múltiple
    st.subheader("Regresión lineal múltiple")
    col1, col2 = st.columns([1,2])
    with col1:
        Variable_y_M = st.selectbox("Variable dependiente (Y)", options=Lista_num, key="rlm_y")
    with col2:
        Variables_x_M = st.multiselect("Variables independientes (X)", options= Lista_num, key="rlm_xs")

    if len(Variables_x_M) >= 1:
        X_M, y_M, droppedM = _clean_xy(numeric_df, Variable_y_M, Variables_x_M)
        if droppedM > 0 and not modo_presentacion:
            st.info(f"Se descartaron {droppedM} filas con NaN/Inf para el ajuste múltiple.")
        if len(y_M) < max(3, len(Variables_x_M)+1):
            st.error("No hay suficientes filas válidas para el modelo múltiple.")
            st.stop()

        Model_M = LinearRegression()
        Model_M.fit(X_M, y_M)
        y_pred_M = Model_M.predict(X_M)

        # Métricas
        coef_Deter_multiple = Model_M.score(X=X_M, y=y_M)
        coef_Correl_multiple = np.sqrt(abs(coef_Deter_multiple))

        # Coeficientes
        coef_tab = pd.DataFrame({
            "Variable": ["Intercepto"] + Variables_x_M,
            "Coeficiente": [Model_M.intercept_] + list(Model_M.coef_)
        })
        if not modo_presentacion:
            st.dataframe(coef_tab, use_container_width=True)

        met_tab = pd.DataFrame({'R^2': [coef_Deter_multiple], 'R ': [coef_Correl_multiple]})
        st.dataframe(met_tab, use_container_width=True)

        # Gráfica: Real vs Predicho
        fig_pred = px.scatter(x=y_M, y=y_pred_M, labels={"x":"Y real ", "y": "Y predicciones"}, title="Comparación Y Real vs Y Predicciones")
        fig_pred.add_trace(go.Scatter(x=[y_M.min(), y_M.max()], y=[y_M.min(), y_M.max()], mode="lines", name="Línea ideal", line=dict(dash="dot")))
        st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.info("Selecciona al menos 1 variable para el modelo múltiple.")


##########################################################################################
# Vista 3
if View == "Regresión No Lineal":
    st.title("Regresión No Lineal")

    # Variables numéricas
    numeric_df = df.select_dtypes(include=['float','float64','int','int64']).copy()
    Lista_num = list(numeric_df.columns)

    contA, contB = st.columns(2)
    with contA:
        Variable_y = st.selectbox("Variable dependiente (Y)", options=Lista_num, key="rnl_y_cf")
    with contB:
        Variable_x = st.selectbox("Variable independiente (X)", options=[c for c in Lista_num if c != Variable_y], key="rnl_x_cf")

    # Modelos disponibles
    modelos = [
        "Función cuadrática (a*x**2 + b*x + c)",
        "Función exponencial (a*np.exp(-b*x)+c)",
        "Función potencia (a*x**b)",
        "Función cúbica (a*x**3 + b*x**2 + c*x + d)"
    ]
    Modelo = st.selectbox("Elige modelo no lineal", options=modelos, key="rnl_modelo_cf")

    # Datos
    df_nl = numeric_df[[Variable_x, Variable_y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(df_nl) < 3:
        st.error("Datos insuficientes tras limpiar NaN/Inf para ajustar el modelo no lineal.")
        st.stop()

    x = df_nl[Variable_x].to_numpy(dtype=float)
    y = df_nl[Variable_y].to_numpy(dtype=float)
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]


    # Definiciones de funciones
    def func_cuad(x, a, b, c):
        return a*x**2 + b*x + c

    def func_cub(x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d

    def func_exp(x, a, b, c):
        return a * np.exp(-b * x) + c

    def func_pot(x, a, b):
        return a * np.power(x, b)

    # Ajuste
    try:
        if Modelo == "Función cuadrática (a*x**2 + b*x + c)":
            pars, cov = curve_fit(func_cuad, x, y, maxfev=20000)
            y_pred = func_cuad(x, *pars)
            y_line = func_cuad(x_sorted, *pars)
            params_df = pd.DataFrame({"Parámetro": ["a", "b", "c"], "Valor": pars})

        elif Modelo == "Función cúbica (a*x**3 + b*x**2 + c*x + d)":
            pars, cov = curve_fit(func_cub, x, y, maxfev=30000)
            y_pred = func_cub(x, *pars)
            y_line = func_cub(x_sorted, *pars)
            params_df = pd.DataFrame({"Parámetro": ["a", "b", "c", "d"], "Valor": pars})

        elif Modelo == "Función exponencial (a*np.exp(-b*x)+c)":
            mask = np.isfinite(y)
            if np.sum(mask) < 3:
                st.error("No hay suficientes datos válidos para ajustar el modelo exponencial.")
                st.stop()
            pars, cov = curve_fit(func_exp, x, y, maxfev=30000)
            y_pred = func_exp(x, *pars)
            y_line = func_exp(x_sorted, *pars)
            params_df = pd.DataFrame({"Parámetro": ["a", "b", "c"], "Valor": pars})

        elif Modelo == "Función potencia (a*x**b)":
            # Requiere x>0 y y>0
            mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3:
                st.error("Para la función potencia se requieren suficientes valores con x>0 e y>0.")
                st.stop()
            x_pos, y_pos = x[mask], y[mask]
            pars, cov = curve_fit(func_pot, x_pos, y_pos, maxfev=20000)
            # Predicciones seguras en todo el rango
            x_safe = np.clip(x, 1e-12, None)
            x_sorted_safe = np.clip(x_sorted, 1e-12, None)
            y_pred = func_pot(x_safe, *pars)
            y_line = func_pot(x_sorted_safe, *pars)
            params_df = pd.DataFrame({"Parámetro": ["a", "b"], "Valor": pars})

        else:
            st.warning("Selecciona un modelo válido.")
            st.stop()

        # Métricas
        r2 = r2_score(y, y_pred)
        r = np.sqrt(abs(r2))

        # Salidas
        st.markdown("**Parámetros estimados (curve_fit):**")
        if not modo_presentacion:
            st.dataframe(params_df, use_container_width=True)

        st.markdown("**Métricas del ajuste:**")
        st.dataframe(pd.DataFrame({"R^2":[r2], "R ":[r]}), use_container_width=True)

        # Gráfica: dispersión + curva predicha
        fig = px.scatter(x=x, y=y, labels={"x": Variable_x, "y": Variable_y},
                         opacity=0.6, title=f"{Modelo} — Dispersión y curva ajustada")
        fig.add_trace(go.Scatter(x=x_sorted, y=y_line, mode="lines", name="Ŷ (curva)", line=dict(width=2)))
        st.plotly_chart(fig, use_container_width=True)

        # Residuos
        resid = y - y_pred
        fig_resid = px.scatter(x=y_pred, y=resid, labels={"x":"Ŷ", "y":"Residual"},
                               title="Residuos vs Predicción")
        fig_resid.add_hline(y=0, line_dash="dot")
        st.plotly_chart(fig_resid, use_container_width=True)

    except RuntimeError as e:
        st.error(f"No convergió el ajuste: {e}. Prueba con otra X/Y o revisa outliers.")
    except Exception as e:
        st.error(f"Error durante el ajuste: {e}")


##########################################################################################
# Vista 4
if View == "Regresión Logística":
    st.title("Regresión Logística")

    # 1) Listas base: Y binaria y X numéricas
    numeric_df = df.select_dtypes(include=['float', 'float64', 'int', 'int64'])
    Lista_num  = list(numeric_df.columns)

    # Detectar dicotómicas (exactamente 2 valores distintos, ignorando NaN)
    dico_cols = []
    for col in df.columns:
        vals = df[col].dropna().unique()
        if len(vals) == 2:
            dico_cols.append(col)

    # Sidebar
    if len(dico_cols) == 0:
        st.warning("No se detectaron variables binarias en el dataset.")
        st.stop()

    Variable_y = st.sidebar.selectbox("Variable dependiente (Y, binaria)", options=dico_cols)
    Variables_x = st.sidebar.multiselect("Variables independientes (X, numéricas)", options=Lista_num)

    # Sliders
    test_size = st.sidebar.slider("Tamaño de prueba", 0.1, 0.5, 0.30, 0.05)
    thr = st.sidebar.slider("Umbral de clasificación", 0.05, 0.95, 0.50, 0.01)

    if len(Variables_x) == 0:
        st.info("Selecciona al menos una variable independiente (X).")
    else:
        # 2) Preparar X, y (sin modificar df original)
        # Trabajar sobre un df que contenga todas las columnas necesarias
        base = df[Variables_x + [Variable_y]].copy()

        # Mapear Y a {0,1} conservando nombres
        vals = base[Variable_y].dropna().unique().tolist()
        if len(vals) != 2:
            st.error(f"La variable '{Variable_y}' debe tener exactamente 2 clases. Encontradas: {vals}")
            st.stop()

        mapping = {vals[0]: 0, vals[1]: 1}
        base['__y__'] = base[Variable_y].map(mapping)

        # Quita NaN/Inf en X y NaN en Y
        base = base.replace([np.inf, -np.inf], np.nan).dropna(subset=Variables_x + ['__y__'])

        if base['__y__'].nunique() < 2:
            st.error("Tras limpiar datos, solo queda una clase en Y. Ajusta la selección de variables o revisa faltantes.")
            st.stop()

        X = base[Variables_x].astype(float).to_numpy()
        y = base['__y__'].to_numpy(dtype=int)
        clases = vals  # para etiquetar métricas

        # 3) Split + escalado        
        # Sidebar extra: manejo de desbalance y estrategia de umbral
        st.sidebar.markdown("### Manejo de desbalance")
        imb_method = st.sidebar.selectbox("Método", ["Ninguno",
                                                    "class_weight='balanced'",
                                                    "SMOTE (over-sampling)",
                                                    "Under-sampling"])

        st.sidebar.markdown("### Estrategia de umbral")
        thr_mode = st.sidebar.selectbox("Seleccionar umbral por…",
                                        ["Manual", "F1 óptimo", "Minimizar costo", "Maximizar recall con precisión mínima"])
        prec_min = None
        c_fp = None
        c_fn = None
        if thr_mode == "Manual":
            thr = st.sidebar.slider("Umbral de clasificación", 0.01, 0.99, thr, 0.01)
        elif thr_mode == "Maximizar recall con precisión mínima":
            prec_min = st.sidebar.slider("Precisión mínima requerida", 0.1, 0.99, 0.6, 0.01)
        elif thr_mode == "Minimizar costo":
            # Ajusta estos valores a tu caso: p. ej., FP=10,000; FN=80,000 (como has usado antes)
            c_fp = st.sidebar.number_input("Costo por FP", min_value=0, value=10000, step=1000)
            c_fn = st.sidebar.number_input("Costo por FN", min_value=0, value=80000, step=1000)

        # 3) Split + escalado
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        escalar = StandardScaler()
        X_train_s = escalar.fit_transform(X_train)
        X_test_s  = escalar.transform(X_test)

        # 3.1) Re-muestreo (solo sobre el set de entrenamiento ya escalado)
        if imb_method == "SMOTE (over-sampling)":
            sm = SMOTE(random_state=42)
            X_train_s, y_train = sm.fit_resample(X_train_s, y_train)
        elif imb_method == "Under-sampling":
            rus = RandomUnderSampler(random_state=42)
            X_train_s, y_train = rus.fit_resample(X_train_s, y_train)

        # 4) Modelo (class_weight según selección)
        if imb_method == "class_weight='balanced'":
            algoritmo = LogisticRegression(max_iter=1000, class_weight='balanced')
        else:
            algoritmo = LogisticRegression(max_iter=1000)

        algoritmo.fit(X_train_s, y_train)

        # 5) Probabilidades y selección de umbral
        y_proba = algoritmo.predict_proba(X_test_s)[:, 1]

        def pick_threshold_by_f1(y_true, y_score):
            p, r, th = precision_recall_curve(y_true, y_score)
            f1 = 2 * (p*r) / np.clip(p+r, 1e-12, None)
            # precision_recall_curve devuelve umbrales len-1 respecto a p/r
            best_idx = np.nanargmax(f1[:-1])
            return th[best_idx], f1[best_idx], p[best_idx], r[best_idx]

        def pick_threshold_by_cost(y_true, y_score, c_fp, c_fn):
            # Recorremos 1001 umbrales uniformes
            ths = np.linspace(0.0, 1.0, 1001)
            best_th, best_cost = 0.5, np.inf
            for t in ths:
                y_pred = (y_score >= t).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                cost = fp * c_fp + fn * c_fn
                if cost < best_cost:
                    best_cost, best_th = cost, t
            return best_th, best_cost

        def pick_threshold_by_recall_with_prec_min(y_true, y_score, prec_min=0.6):
            p, r, th = precision_recall_curve(y_true, y_score)
            # p/r len N, th len N-1. Usamos índices de th.
            valid = np.where(p[:-1] >= prec_min)[0]
            if len(valid) == 0:
                return 0.5, 0.0, 0.0  # fallback
            # entre los que cumplen precisión mínima, elegimos el de mayor recall
            best_idx = valid[np.argmax(r[valid])]
            return th[best_idx], r[best_idx], p[best_idx]

        # Elegimos umbral según estrategia
        if thr_mode == "F1 óptimo":
            thr, best_f1, best_p, best_r = pick_threshold_by_f1(y_test, y_proba)
        elif thr_mode == "Minimizar costo":
            thr, best_cost = pick_threshold_by_cost(y_test, y_proba, c_fp, c_fn)
        elif thr_mode == "Maximizar recall con precisión mínima":
            thr, best_r, best_p = pick_threshold_by_recall_with_prec_min(y_test, y_proba, prec_min=prec_min)
        # Si es "Manual", ya viene de la sidebar

        y_pred = (y_proba >= thr).astype(int)

        # 6) Métricas ampliadas
        acc     = accuracy_score(y_test, y_pred)
        bacc    = balanced_accuracy_score(y_test, y_pred)
        prec_c0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
        prec_c1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec_c0  = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        rec_c1  = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_min  = f1_score(y_test, y_pred, pos_label=1, zero_division=0)  # F1 de la minoritaria (etiqueta 1)
        auc     = roc_auc_score(y_test, y_proba)
        auprc   = average_precision_score(y_test, y_proba)  # área bajo curva Prec-Recall (clase 1)

        # Tabla de métricas
        met_rows = [
            ("Exactitud", acc),
            ("Balanced accuracy", bacc),
            (f"Precision ({clases[0]})", prec_c0),
            (f"Precision ({clases[1]})", prec_c1),
            (f"Sensibilidad ({clases[0]})", rec_c0),
            (f"Sensibilidad ({clases[1]})", rec_c1),
            (f"F1 ({clases[1]})", f1_min),
            ("ROC-AUC", auc)
        ]
        if thr_mode == "Minimizar costo":
            met_rows.append(("Costo total (FP/FN)", best_cost))

        met_tab = pd.DataFrame(met_rows, columns=["Métrica", "Valor"])
        st.subheader("Métricas")
        st.dataframe(met_tab, use_container_width=True)

        # Alertas útiles
        prev = y_test.mean()
        if prec_c1 == 1.0 and rec_c1 < 0.15:
            st.warning("La precisión de la clase minoritaria es 1.0 pero el recall es muy bajo. "
                    "Baja el umbral, usa class_weight='balanced' o aplica re-muestreo.")
        if acc > 0.9 and bacc < 0.65 and prev < 0.25:
            st.info("La exactitud es alta por el desbalance. Revisa balanced accuracy, AUPRC y F1 de la minoritaria.")

        # 7) Coeficientes y Odds Ratios (sin cambios)
        coef = algoritmo.coef_[0]
        intercepto = algoritmo.intercept_[0]
        coef_tab = pd.DataFrame({
            "Variable": ["Intercepto"] + Variables_x,
            "Coeficiente (log-odds)": [intercepto] + list(coef),
            "Odds Ratio (exp(coef))": [np.exp(intercepto)] + list(np.exp(coef))
        })
        if not modo_presentacion:
            st.subheader("Coeficientes del modelo")
            st.dataframe(coef_tab, use_container_width=True)

        # 8) Matriz de confusión (igual que ya tenías)
        matriz = confusion_matrix(y_test, y_pred, labels=[0, 1])
        labels_disp = [clases[0], clases[1]]
        fig_cm = go.Figure(data=go.Heatmap(
            z=matriz,
            x=[f"Pred {labels_disp[0]}", f"Pred {labels_disp[1]}"],
            y=[f"Real {labels_disp[0]}", f"Real {labels_disp[1]}"],
            colorscale="Oranges", showscale=True, hoverongaps=False
        ))
        ann = []
        tags = np.array([["TN","FP"],["FN","TP"]])
        for i in range(2):
            for j in range(2):
                ann.append(dict(
                    x=[f"Pred {labels_disp[0]}", f"Pred {labels_disp[1]}"][j],
                    y=[f"Real {labels_disp[0]}", f"Real {labels_disp[1]}"][i],
                    text=f"{tags[i,j]}: {matriz[i,j]}",
                    showarrow=False,
                    font=dict(color="white" if matriz[i,j] > matriz.max()/2 else "black")
                ))
        fig_cm.update_layout(title="Matriz de confusión", annotations=ann, width=520, height=520)
        st.plotly_chart(fig_cm, use_container_width=False)

        # 9) Curva ROC (igual)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Aleatorio", line=dict(dash="dot")))
        fig_roc.update_layout(title="Curva ROC", xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig_roc, use_container_width=True)

        # 10) Curva Precisión-Recall y distribución de probabilidades
        p, r, th = precision_recall_curve(y_test, y_proba)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=r, y=p, mode="lines", name=f"PR (AP={auprc:.3f})"))
        fig_pr.update_layout(title="Curva Precisión-Recall (clase 1)",
                            xaxis_title="Recall", yaxis_title="Precisión")
        st.plotly_chart(fig_pr, use_container_width=True)

        fig_prob = px.strip(
            x=[labels_disp[i] for i in y_test], y=y_proba,
            labels={"x":"Clase real", "y":"Probabilidad P(Y=1)"},
            title="Distribución de probabilidades por clase real"
        )
        fig_prob.add_hline(y=thr, line_dash="dot", annotation_text=f"Umbral {thr:.2f}")
        st.plotly_chart(fig_prob, use_container_width=True)

        # Nota de mapeo (como ya tenías)
        st.caption(f"Mapeo interno (solo para el modelo): {clases[0]} → 0, {clases[1]} → 1. "
                f"Prevalencia clase 1 (test): {prev:.3f}")

# FOOTER
st.markdown("---")
st.markdown("""
<div style="text-align:center; opacity:0.8; font-size:0.9rem;">
© Proyecto para Gestión de Proyectos — Dashboard creado por <b>Raymundo Díaz</b> con ayuda de IA y profe Freddy.  
<br> Construido con Streamlit, Plotly y Python.
</div>
""", unsafe_allow_html=True)

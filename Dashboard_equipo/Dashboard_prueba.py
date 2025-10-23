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
    df = df.drop(['Unnamed: 0','latitude','longitude'], axis=1, errors="ignore")
    if 'host_id' in df.columns:
        df['host_id'] = df['host_id'].astype(str)

    for col in ['host_is_superhost','host_identity_verified','instant_bookable']:
        if col in df.columns:
            df[col] = _normalize_binary(df[col])

    for col in ['host_response_rate','host_acceptance_rate','price']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

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
        superhosts = (df['host_is_superhost'] == '1').sum() if 'host_is_superhost' in df.columns else 0
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
# (Las otras vistas: Lineal, No Lineal, Logística y Comparar países se mantienen igual)
##########################################################################################

# FOOTER
st.markdown("---")
st.markdown("""
<div style="text-align:center; opacity:0.8; font-size:0.9rem;">
© Proyecto para Gestión de Proyectos — Dashboard creado por <b>Raymundo Díaz</b> con ayuda de IA y profe Freddy.  
<br> Construido con Streamlit, Plotly y Python.
</div>
""", unsafe_allow_html=True)

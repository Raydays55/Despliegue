# Dashboard Final equipo — Proyecto Airbnb (By Raymundo Díaz + IA + Profe Freddy)
# Versión final optimizada y multi-país comparativo

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
html, body, [data-testid="stAppViewContainer"], section[data-testid="stSidebar"] {{
    background: radial-gradient(circle at 30% 30%, #131722 0%, #0E1117 100%) !important;
    color: white !important;
}}
section[data-testid="stSidebar"] {{ border-right: 1px solid {AIRBNB_BORDER}; }}
.air-card {{
    border: 1px solid {AIRBNB_BORDER};
    border-radius:16px; padding:1rem;
    background:{AIRBNB_CARD};
}}
.stButton>button {{
    background:{AIRBNB_RED}; color:white; border-radius:12px; border:none;
    padding:.6rem 1rem; font-weight:600;
}}
.stButton>button:hover {{ opacity:.9 }}
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
# Normalización y carga base
BIN_TRUE = {"t","true","True",1,"1",True}
BIN_FALSE= {"f","false","False",0,"0",False}

def _normalize_binary(series):
    s = series.copy()
    return s.apply(lambda v: 1 if v in BIN_TRUE else (0 if v in BIN_FALSE else np.nan)).astype("float")

def _normalize_df(df_raw):
    df = df_raw.copy()
    df = df.drop(['Unnamed: 0','latitude','longitude'], axis=1, errors="ignore")
    if 'id' in df.columns:
        df['id'] = df['id'].astype(str)
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
    raw = pd.read_csv(COUNTRY_FILES[country])
    df = _normalize_df(raw)
    Lista = [
        'host_is_superhost','host_identity_verified','host_response_time',
        'host_response_rate','host_acceptance_rate','host_total_listings_count',
        'host_verifications','room_type','property_type','price_cat'
    ]
    return df, Lista

#  Helpers multi-país 
@st.cache_data(show_spinner=False)
def load_all_countries():
    data = {}
    for c in COUNTRY_FILES.keys():
        df_i, lista_i = load_country_df(c)
        data[c] = {"df": df_i, "lista": lista_i}
    return data

def _top_freq(df, var_cat, k=10):
    if var_cat not in df.columns: return None
    tab = df[var_cat].value_counts(dropna=False).reset_index().head(k)
    tab.columns = ["categorias","frecuencia"]
    return tab

def _grid_2x2_figs(figs, titles):
    cols = st.columns(2)
    for i, (fig, title) in enumerate(zip(figs, titles)):
        with cols[i % 2]:
            if title: st.caption(f"**{title}**")
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No disponible para este país.")

def _numdf(df): return df.select_dtypes(include=["float","int"]).copy()

##########
# Header
col_logo, col_title = st.columns([1,5], vertical_alignment="center")
with col_logo:
    st.image("assets/Logo3.jpg", width=90)
with col_title:
    st.markdown("""
        # Airbnb Data Analysis — Multi-país
        <span style="color:#767676">Comparativa entre Alemania, Valencia, Estocolmo y México</span>
    """, unsafe_allow_html=True)

##########
# Sidebar
st.sidebar.image("assets/Logoo.jpg", use_container_width=True)
st.sidebar.caption("Análisis exploratorio y modelos comparativos")
st.sidebar.markdown("---")

modo_presentacion = st.sidebar.toggle("Modo presentación", value=False)
View = st.sidebar.selectbox(
    label='Tipo de análisis',
    options=['Extracción de Características', 'Regresión Lineal', 'Regresión No Lineal', 'Regresión Logística'],
    index=0
)

##########################################################################################
# VISTA 1 — EXTRACCIÓN DE CARACTERÍSTICAS
if View == "Extracción de Características":
    all_data = load_all_countries()

    st.title("Extracción de Características — 4 países")
    st.caption("Comparación de distribuciones categóricas entre países Airbnb.")

    Variable_Cat = st.sidebar.selectbox("Variable categórica a analizar", options=list(all_data["Alemania"]["lista"]))
    k_top = st.sidebar.slider("Top categorías a mostrar", 5, 20, 10, 1)

    # --- Distribución por categoría (Bar) ---
    st.subheader("Distribución por categoría (Bar Plot)")
    bar_figs, titles = [], []
    for c, pack in all_data.items():
        df_i = pack["df"]
        tab = _top_freq(df_i, Variable_Cat, k=k_top)
        if tab is not None and len(tab) > 0:
            fig = px.bar(tab, x="categorias", y="frecuencia", color="categorias")
        else:
            fig = None
        bar_figs.append(fig); titles.append(c)
    _grid_2x2_figs(bar_figs, titles)

    # --- Pie Chart ---
    st.subheader("Proporción por categoría (Pie Chart)")
    pie_figs, titles = [], []
    for c, pack in all_data.items():
        df_i = pack["df"]
        tab = _top_freq(df_i, Variable_Cat, k=k_top)
        if tab is not None:
            fig = px.pie(tab, names="categorias", values="frecuencia")
        else:
            fig = None
        pie_figs.append(fig); titles.append(c)
    _grid_2x2_figs(pie_figs, titles)

    # --- Donut Chart ---
    st.subheader("Visualización tipo dona")
    donut_figs, titles = [], []
    for c, pack in all_data.items():
        df_i = pack["df"]
        tab = _top_freq(df_i, Variable_Cat, k=k_top)
        if tab is not None:
            fig = px.pie(tab, names="categorias", values="frecuencia", hole=0.5)
        else:
            fig = None
        donut_figs.append(fig); titles.append(c)
    _grid_2x2_figs(donut_figs, titles)

    # --- Área acumulada ---
    st.subheader("Tendencia acumulada (Área)")
    area_figs, titles = [], []
    for c, pack in all_data.items():
        df_i = pack["df"]
        tab = _top_freq(df_i, Variable_Cat, k=k_top)
        if tab is not None:
            tab_sorted = tab.sort_values(by="frecuencia", ascending=False)
            fig = px.area(tab_sorted, x="categorias", y="frecuencia")
        else:
            fig = None
        area_figs.append(fig); titles.append(c)
    _grid_2x2_figs(area_figs, titles)

    # --- Análisis más profundo ---
    st.markdown("---")
    st.subheader("Análisis más profundo (Boxplot/Heatmap)")
    deep_figs, titles = [], []
    for c, pack in all_data.items():
        df_i = pack["df"]
        if Variable_Cat in ['room_type','property_type','price_cat'] and 'price' in df_i.columns:
            if not df_i[[Variable_Cat,'price']].dropna().empty:
                fig = px.box(df_i, x=Variable_Cat, y='price', color=Variable_Cat)
            else:
                fig = None
        else:
            heat_df = pd.crosstab(index=df_i[Variable_Cat], columns='count', normalize='columns') * 100
            fig = px.imshow(heat_df, color_continuous_scale=CONT_GRADIENT) if not heat_df.empty else None
        deep_figs.append(fig); titles.append(c)
    _grid_2x2_figs(deep_figs, titles)

##########################################################################################
# ======== VISTA 2 — REGRESIÓN LINEAL MULTI-PAÍS
if View == "Regresión Lineal":
    all_data = load_all_countries()
    st.title("Regresión Lineal — 4 países")

    num_cols = list(_numdf(all_data["Alemania"]["df"]).columns)
    colL, colR = st.columns(2)
    with colL:
        Variable_y = st.selectbox("Variable dependiente (Y)", options=num_cols, key="lin_y")
    with colR:
        Variable_x = st.selectbox("Variable independiente (X)", options=num_cols, key="lin_x")

    st.subheader("Dispersión + recta ajustada (Y ~ X)")
    fig_list, title_list, rows = [], [], []
    for c, pack in all_data.items():
        df_i = _numdf(pack["df"]).dropna(subset=[Variable_x, Variable_y])
        if len(df_i) < 10:
            fig, met = None, None
        else:
            X = df_i[[Variable_x]].values; y = df_i[Variable_y].values
            mdl = LinearRegression().fit(X, y)
            yhat = mdl.predict(X)
            fig = px.scatter(df_i, x=Variable_x, y=Variable_y, opacity=0.6, title=f"{c}")
            fig.add_trace(go.Scatter(x=np.sort(X[:,0]), y=np.sort(yhat), mode="lines", name="Ajuste"))
            met = {"R2": mdl.score(X,y), "R": np.sqrt(abs(mdl.score(X,y))),
                   "Coef": mdl.coef_[0], "Intercepto": mdl.intercept_}
        fig_list.append(fig); title_list.append(c)
        if met: rows.append({"País": c, **met})
        else: rows.append({"País": c, "R2": np.nan, "R": np.nan, "Coef": np.nan, "Intercepto": np.nan})

    _grid_2x2_figs(fig_list, title_list)
    st.markdown("**Comparativa de métricas:**")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

##########################################################################################
# VISTA 3 — REGRESIÓN NO LINEAL MULTI-PAÍS
if View == "Regresión No Lineal":
    all_data = load_all_countries()
    st.title("Regresión No Lineal — 4 países")

    num_cols = list(_numdf(all_data["Alemania"]["df"]).columns)
    Variable_y = st.selectbox("Variable dependiente (Y)", options=num_cols)
    Variable_x = st.selectbox("Variable independiente (X)", options=[c for c in num_cols if c != Variable_y])
    Modelo = st.selectbox("Modelo no lineal", options=[
        "Función cuadrática (a*x**2 + b*x + c)",
        "Función cúbica (a*x**3 + b*x**2 + c*x + d)",
        "Función exponencial (a*np.exp(-b*x)+c)",
        "Función potencia (a*x**b)"
    ])

    def fit_curve(df, x, y, model):
        x_, y_ = df[x].dropna(), df[y].dropna()
        if len(x_) < 15 or len(y_) < 15:
            return None, None
        x_ = x_.to_numpy(); y_ = y_.to_numpy()
        if model.startswith("Función cuadrática"):
            def f(x,a,b,c): return a*x**2+b*x+c
        elif model.startswith("Función cúbica"):
            def f(x,a,b,c,d): return a*x**3+b*x**2+c*x+d
        elif model.startswith("Función exponencial"):
            def f(x,a,b,c): return a*np.exp(-b*x)+c
        else:
            def f(x,a,b): return a*np.power(x,b)
        try:
            pars,_=curve_fit(f,x_,y_,maxfev=20000)
            yhat=f(x_,*pars)
            r2=r2_score(y_,yhat)
            fig=px.scatter(df,x=x,y=y,opacity=0.6)
            fig.add_trace(go.Scatter(x=np.sort(x_),y=f(np.sort(x_),*pars),mode="lines"))
            return fig,{"R2":r2,"R":np.sqrt(abs(r2))}
        except:
            return None,None

    figs, titles, rows = [], [], []
    for c, pack in all_data.items():
        df_i = _numdf(pack["df"])
        fig, met = fit_curve(df_i, Variable_x, Variable_y, Modelo)
        figs.append(fig); titles.append(c)
        if met: rows.append({"País": c, **met})
        else: rows.append({"País": c, "R2": np.nan, "R": np.nan})
    _grid_2x2_figs(figs, titles)
    st.markdown("**Métricas comparadas:**")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

##########################################################################################
# VISTA 4 — REGRESIÓN LOGÍSTICA MULTI-PAÍS
##########################################################################################
if View == "Regresión Logística":
    all_data = load_all_countries()
    st.title("Regresión Logística — 4 países")

    d0 = all_data["Alemania"]["df"]
    dico_cols = [c for c in d0.columns if d0[c].dropna().nunique() == 2]
    num_cols = list(_numdf(d0).columns)

    Variable_y = st.sidebar.selectbox("Variable dependiente (binaria)", options=dico_cols)
    Variables_x = st.sidebar.multiselect("Variables independientes (numéricas)", options=num_cols)
    test_size = st.sidebar.slider("Tamaño de prueba", 0.1, 0.5, 0.3)
    thr = st.sidebar.slider("Umbral", 0.05, 0.95, 0.5)
    imb_method = st.sidebar.selectbox("Balance de clases", ["Ninguno", "SMOTE", "Under", "class_weight='balanced'"])

    st.subheader("Métricas por país")
    rows = []
    for c, pack in all_data.items():
        df_i = pack["df"]
        if Variable_y not in df_i.columns or len(Variables_x)==0:
            rows.append({"País":c,"AUC":np.nan,"Balanced Acc":np.nan,"F1":np.nan}); continue
        df_i = df_i.dropna(subset=[Variable_y]+Variables_x)
        if df_i.empty: 
            rows.append({"País":c,"AUC":np.nan,"Balanced Acc":np.nan,"F1":np.nan}); continue

        clases = df_i[Variable_y].dropna().unique().tolist()
        if len(clases)!=2: continue
        y = df_i[Variable_y].map({clases[0]:0,clases[1]:1}).values
        X = df_i[Variables_x].astype(float).values
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,stratify=y,random_state=42)
        sc=StandardScaler(); X_train=sc.fit_transform(X_train); X_test=sc.transform(X_test)

        if imb_method=="SMOTE": X_train,y_train=SMOTE(random_state=42).fit_resample(X_train,y_train)
        if imb_method=="Under": X_train,y_train=RandomUnderSampler(random_state=42).fit_resample(X_train,y_train)

        model=LogisticRegression(max_iter=1000,class_weight=('balanced' if imb_method=="class_weight='balanced'" else None))
        model.fit(X_train,y_train)
        proba=model.predict_proba(X_test)[:,1]
        pred=(proba>=thr).astype(int)

        auc=roc_auc_score(y_test,proba)
        bacc=balanced_accuracy_score(y_test,pred)
        f1m=f1_score(y_test,pred,zero_division=0)
        rows.append({"País":c,"AUC":auc,"Balanced Acc":bacc,"F1":f1m})

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

##########################################################################################
# FOOTER
##########################################################################################
st.markdown("---")
st.markdown("""
<div style="text-align:center; opacity:0.8; font-size:0.9rem;">
© Proyecto para Gestión de Proyectos — Dashboard creado por <b>Raymundo Díaz</b> con ayuda de IA y profe Freddy.  
<br>Comparativo multi-país de Airbnb con Streamlit y Plotly.
</div>
""", unsafe_allow_html=True)


# Dashboard Final equipo integrando en las vistas a los 4 países - Proyecto Airbnb
# Versión final

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
# Normalización
BIN_TRUE = {"t","true","True",1,"1",True}
BIN_FALSE= {"f","false","False",0,"0",False}

def _normalize_binary(series):
    s = series.copy()
    return s.apply(lambda v: 1 if v in BIN_TRUE else (0 if v in BIN_FALSE else np.nan)).astype("float")

def _normalize_df(df_raw):
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)]
    df = df.drop(['latitude','longitude','first_review','last_review','host_since', 'price', 'estimated_revenue_l365d','source','id', 'scrape_id'],
                 axis=1, errors="ignore")
    if 'id' in df.columns:
        df['id'] = df['id'].astype(str)
    if 'host_id' in df.columns:
        df['host_id'] = df['host_id'].astype(str)
    for col in ['host_is_superhost','host_identity_verified','instant_bookable']:
        if col in df.columns:
            df[col] = _normalize_binary(df[col])
    for col in ['host_response_rate','host_acceptance_rate','price','estimated_revenue_l365d','price_eur']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def _clean_xy(df_base, y_col, x_cols):
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

def kpis_block(df, country):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="air-card">', unsafe_allow_html=True)
        st.metric(f"{country} · Filas", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="air-card">', unsafe_allow_html=True)
        st.metric(f"{country} · Tipos de propiedad", df['property_type'].nunique() if 'property_type' in df.columns else "—")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="air-card">', unsafe_allow_html=True)
        med_price = np.nanmean(df['price_eur']) if 'price_eur' in df.columns else np.nan
        st.metric(f"{country} · Media precio", f"€{med_price:,.0f}" if np.isfinite(med_price) else "—")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="air-card">', unsafe_allow_html=True)
        superhosts = int((df['host_is_superhost'] == 1).sum()) if 'host_is_superhost' in df.columns else 0
        st.metric(f"{country} · Superhosts", superhosts)
        st.markdown('</div>', unsafe_allow_html=True)

def extraction_charts(df, var_cat: str):
    tabla = df[var_cat].value_counts(dropna=False).reset_index().head(10)
    tabla.columns = ['categorias','frecuencia']
    fig_bar = px.bar(tabla, x='categorias', y='frecuencia', color='categorias', title="Distribución por categoría")
    fig_pie = px.pie(tabla, names='categorias', values='frecuencia', title="Proporción por categoría")
    fig_donut = px.pie(tabla, names='categorias', values='frecuencia', hole=0.5, title="Gráfico tipo anillo")
    fig_area = px.area(tabla.sort_values('frecuencia', ascending=False),
                       x='categorias', y='frecuencia', title="Tendencia acumulada (Área)")
    # Detalle: box/heatmap según exista price
    detail_fig = None
    if var_cat in ['room_type','property_type','price_cat'] and 'price' in df.columns:
        detail_fig = px.box(df, x=var_cat, y='price', color=var_cat, title="Relación categorías vs precio (Boxplot)")
    else:
        heat_df = pd.crosstab(index=df[var_cat], columns='count', normalize='columns') * 100
        detail_fig = px.imshow(heat_df, color_continuous_scale=CONT_GRADIENT, title="Proporción por categoría (Heatmap)")
    return tabla, fig_bar, fig_pie, fig_donut, fig_area, detail_fig

def gallery_block(country):
    st.markdown(f"**Galería:** {country} — Airbnb")
    imgs = COUNTRY_IMAGES.get(country, [])
    gcols = st.columns(3)
    for i, path in enumerate(imgs[:3]):
        with gcols[i]:
            try:
                st.image(path, use_container_width=True)
            except Exception:
                st.write("🖼️ Imagen no encontrada")

def get_common_lists(dfs_dict):
    # Intersección de columnas numéricas y binarias para logística / extracción
    num_sets = []
    bin_sets = []
    cat_sets = []
    for _, df in dfs_dict.items():
        num_cols = set(df.select_dtypes(include=['float','float64','int','int64']).columns.tolist())
        # binarias: exactamente 2 valores (ignorando NaN)
        bin_cols = set([c for c in df.columns if df[c].dropna().nunique()==2])
        # categóricas candidatas (object o categóricas + algunas conocidas)
        cat_cols = set([c for c in df.columns if df[c].dtype=='object' or df[c].dtype.name=='category'])
        # agrega columnas 'conocidas' aunque sean numéricas codificadas
        cat_cols |= set([c for c in ['room_type','property_type','price_cat','host_response_time'] if c in df.columns])
        num_sets.append(num_cols)
        bin_sets.append(bin_cols)
        cat_sets.append(cat_cols)
    common_num = set.intersection(*num_sets) if num_sets else set()
    common_bin = set.intersection(*bin_sets) if bin_sets else set()
    common_cat = set.intersection(*cat_sets) if cat_sets else set()
    # excluir target obvios de num si molestan
    return sorted(list(common_num)), sorted(list(common_bin)), sorted(list(common_cat))

def run_logistic_block(df, y_col, x_cols, thr_mode="Manual", thr=0.5, c_fp=10000, c_fn=80000, prec_min=0.6, test_size=0.30, imb_method="Ninguno"):
    base = df[x_cols + [y_col]].copy()
    vals = base[y_col].dropna().unique().tolist()
    if len(vals) != 2:
        return None
    mapping = {vals[0]:0, vals[1]:1}
    base['__y__'] = base[y_col].map(mapping)
    base = base.replace([np.inf,-np.inf], np.nan).dropna(subset=x_cols + ['__y__'])
    if base['__y__'].nunique() < 2:
        return None
    X = base[x_cols].astype(float).to_numpy()
    y = base['__y__'].to_numpy(dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    if imb_method == "SMOTE (over-sampling)":
        sm = SMOTE(random_state=42); X_train_s, y_train = sm.fit_resample(X_train_s, y_train)
    elif imb_method == "Under-sampling":
        rus = RandomUnderSampler(random_state=42); X_train_s, y_train = rus.fit_resample(X_train_s, y_train)
    if imb_method == "class_weight='balanced'":
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    else:
        clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_s, y_train)
    y_proba = clf.predict_proba(X_test_s)[:,1]

    def pick_threshold_by_f1(y_true, y_score):
        p, r, th = precision_recall_curve(y_true, y_score)
        f1 = 2*(p*r)/np.clip(p+r, 1e-12, None)
        best_idx = np.nanargmax(f1[:-1])
        return th[best_idx]
    def pick_threshold_by_cost(y_true, y_score, c_fp, c_fn):
        ths = np.linspace(0.0,1.0,1001)
        best_th, best_cost = 0.5, np.inf
        for t in ths:
            y_pred = (y_score>=t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            cost = fp*c_fp + fn*c_fn
            if cost < best_cost:
                best_cost, best_th = cost, t
        return best_th
    def pick_threshold_by_recall_with_prec_min(y_true, y_score, prec_min=0.6):
        p, r, th = precision_recall_curve(y_true, y_score)
        valid = np.where(p[:-1] >= prec_min)[0]
        if len(valid)==0: return 0.5
        best_idx = valid[np.argmax(r[valid])]
        return th[best_idx]

    if thr_mode=="F1 óptimo":
        thr = pick_threshold_by_f1(y_test, y_proba)
    elif thr_mode=="Minimizar costo":
        thr = pick_threshold_by_cost(y_test, y_proba, c_fp, c_fn)
    elif thr_mode=="Maximizar recall con precisión mínima":
        thr = pick_threshold_by_recall_with_prec_min(y_test, y_proba, prec_min=prec_min)
    # Manual: se respeta valor de thr

    y_pred = (y_proba>=thr).astype(int)
    acc   = accuracy_score(y_test, y_pred)
    bacc  = balanced_accuracy_score(y_test, y_pred)
    prec1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec1  = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1m   = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    auc   = roc_auc_score(y_test, y_proba)
    auprc = average_precision_score(y_test, y_proba)
    cm    = confusion_matrix(y_test, y_pred, labels=[0,1])

    # Figuras
    labels_disp = [list(mapping.keys())[0], list(mapping.keys())[1]]
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
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
                text=f"{tags[i,j]}: {cm[i,j]}",
                showarrow=False,
                font=dict(color="white" if cm[i,j] > cm.max()/2 else "black")
            ))
    fig_cm.update_layout(title=f"Matriz de confusión · umbral={thr:.2f}", annotations=ann, width=520, height=520)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Aleatorio", line=dict(dash="dot")))
    fig_roc.update_layout(title="Curva ROC", xaxis_title="FPR", yaxis_title="TPR")

    p, r, th = precision_recall_curve(y_test, y_proba)
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=r, y=p, mode="lines", name=f"PR (AP={auprc:.3f})"))
    fig_pr.update_layout(title="Curva Precisión-Recall (clase 1)", xaxis_title="Recall", yaxis_title="Precisión")

    fig_prob = px.strip(
        x=[labels_disp[i] for i in y_test], y=y_proba,
        labels={"x":"Clase real", "y":"Probabilidad P(Y=1)"},
        title="Distribución de probabilidades por clase real"
    )
    fig_prob.add_hline(y=thr, line_dash="dot", annotation_text=f"Umbral {thr:.2f}")

    met_tab = pd.DataFrame({
        "Métrica": ["Exactitud","Balanced accuracy","Precisión (1)","Recall (1)","F1 (1)","ROC-AUC","AP (PR)"],
        "Valor":   [acc, bacc, prec1, rec1, f1m, auc, auprc]
    })
    return dict(
        metrics=met_tab, cm_fig=fig_cm, roc_fig=fig_roc, pr_fig=fig_pr, prob_fig=fig_prob,
        thr=thr, mapping=mapping
    )

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
        med_price = np.nanmean(df['price_eur']) if 'price_eur' in df.columns else np.nan
        st.metric("Media de precio", f"€{med_price:,.0f}" if np.isfinite(med_price) else "—")
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

    gallery_block(country)

##########################################################################################
# Vista 2 — Regresión Lineal
if View == "Regresión Lineal":
    st.title("Regresión Lineal")

    numeric_df = df.select_dtypes(include=['float', 'float64', 'int', 'int64']).copy()
    Lista_num = list(numeric_df.columns)

    st.subheader("Regresión lineal simple")
    colL, colR = st.columns(2)
    with colL:
        Variable_y = st.selectbox("Variable dependiente (Y)", options=Lista_num, key="rl_y")
    with colR:
        Variable_x = st.selectbox("Variable independiente (X)", options=Lista_num, key="rl_x")

    X, y, dropped = _clean_xy(numeric_df, Variable_y, [Variable_x])
    if dropped > 0 and not modo_presentacion:
        st.info(f"Se descartaron {dropped} filas con NaN/Inf para el ajuste.")
    if len(y) < 3:
        st.error("No hay suficientes filas válidas para ajustar el modelo.")
        st.stop()

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    coef_Deter_simple = model.score(X= X, y= y)
    coef_Correl_simple = np.sqrt(abs(coef_Deter_simple))

    coef_df_simple = pd.DataFrame({
        "Variable": [Variable_x],
        "Coeficiente": [model.coef_[0]],
        "Intercepto": [model.intercept_],
        "R": [coef_Correl_simple],
        "R^2": [coef_Deter_simple]
    })
    if not modo_presentacion:
        st.dataframe(coef_df_simple, use_container_width=True)

    fig_scat = px.scatter(numeric_df, x=Variable_x, y=Variable_y, opacity=0.6, title="Dispersión y recta ajustada")
    order_idx = np.argsort(X[:, 0])
    fig_scat.add_trace(go.Scatter(
        x=X[order_idx, 0], y=y_pred[order_idx],
        mode="lines", name="Predicción de Y"
    ))
    st.plotly_chart(fig_scat, use_container_width=True)

    resid = y - y_pred
    fig_res = px.scatter(x=y_pred, y=resid, labels={"x":"Ŷ", "y":"Residual"},
                         title="Residuos vs Predicción (diagnóstico)")
    fig_res.add_hline(y=0, line_dash="dot")
    st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("---")
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

        coef_Deter_multiple = Model_M.score(X=X_M, y=y_M)
        coef_Correl_multiple = np.sqrt(abs(coef_Deter_multiple))

        coef_tab = pd.DataFrame({
            "Variable": ["Intercepto"] + Variables_x_M,
            "Coeficiente": [Model_M.intercept_] + list(Model_M.coef_)
        })
        if not modo_presentacion:
            st.dataframe(coef_tab, use_container_width=True)

        met_tab = pd.DataFrame({'R^2': [coef_Deter_multiple], 'R ': [coef_Correl_multiple]})
        st.dataframe(met_tab, use_container_width=True)

        fig_pred = px.scatter(x=y_M, y=y_pred_M, labels={"x":"Y real ", "y": "Y predicciones"}, title="Comparación Y Real vs Y Predicciones")
        fig_pred.add_trace(go.Scatter(x=[y_M.min(), y_M.max()], y=[y_M.min(), y_M.max()], mode="lines", name="Línea ideal", line=dict(dash="dot")))
        st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.info("Selecciona al menos 1 variable para el modelo múltiple.")

##########################################################################################
# Vista 3 — Regresión No Lineal
if View == "Regresión No Lineal":
    st.title("Regresión No Lineal")

    numeric_df = df.select_dtypes(include=['float','float64','int','int64']).copy()
    Lista_num = list(numeric_df.columns)

    contA, contB = st.columns(2)
    with contA:
        Variable_y = st.selectbox("Variable dependiente (Y)", options=Lista_num, key="rnl_y_cf")
    with contB:
        Variable_x = st.selectbox("Variable independiente (X)", options=[c for c in Lista_num if c != Variable_y], key="rnl_x_cf")

    modelos = [
        "Función cuadrática (a*x**2 + b*x + c)",
        "Función exponencial (a*np.exp(-b*x)+c)",
        "Función potencia (a*x**b)",
        "Función cúbica (a*x**3 + b*x**2 + c*x + d)"
    ]
    Modelo = st.selectbox("Elige modelo no lineal", options=modelos, key="rnl_modelo_cf")

    df_nl = numeric_df[[Variable_x, Variable_y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(df_nl) < 3:
        st.error("Datos insuficientes tras limpiar NaN/Inf para ajustar el modelo no lineal.")
        st.stop()

    x = df_nl[Variable_x].to_numpy(dtype=float)
    y = df_nl[Variable_y].to_numpy(dtype=float)
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]

    def func_cuad(x, a, b, c): return a*x**2 + b*x + c
    def func_cub(x, a, b, c, d): return a*x**3 + b*x**2 + c*x + d
    def func_exp(x, a, b, c): return a * np.exp(-b * x) + c
    def func_pot(x, a, b): return a * np.power(x, b)

    try:
        if Modelo == "Función cuadrática (a*x**2 + b*x + c)":
            pars, cov = curve_fit(func_cuad, x, y, maxfev=20000)
            y_pred = func_cuad(x, *pars); y_line = func_cuad(x_sorted, *pars)
            params_df = pd.DataFrame({"Parámetro": ["a", "b", "c"], "Valor": pars})
        elif Modelo == "Función cúbica (a*x**3 + b*x**2 + c*x + d)":
            pars, cov = curve_fit(func_cub, x, y, maxfev=30000)
            y_pred = func_cub(x, *pars); y_line = func_cub(x_sorted, *pars)
            params_df = pd.DataFrame({"Parámetro": ["a", "b", "c", "d"], "Valor": pars})
        elif Modelo == "Función exponencial (a*np.exp(-b*x)+c)":
            mask = np.isfinite(y)
            if np.sum(mask) < 3: st.error("No hay suficientes datos válidos para el modelo exponencial."); st.stop()
            pars, cov = curve_fit(func_exp, x, y, maxfev=30000)
            y_pred = func_exp(x, *pars); y_line = func_exp(x_sorted, *pars)
            params_df = pd.DataFrame({"Parámetro": ["a", "b", "c"], "Valor": pars})
        elif Modelo == "Función potencia (a*x**b)":
            mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3: st.error("Para potencia se requieren suficientes valores con x>0 e y>0."); st.stop()
            x_pos, y_pos = x[mask], y[mask]
            pars, cov = curve_fit(func_pot, x_pos, y_pos, maxfev=20000)
            x_safe = np.clip(x, 1e-12, None); x_sorted_safe = np.clip(x_sorted, 1e-12, None)
            y_pred = func_pot(x_safe, *pars); y_line = func_pot(x_sorted_safe, *pars)
            params_df = pd.DataFrame({"Parámetro": ["a", "b"], "Valor": pars})
        else:
            st.warning("Selecciona un modelo válido."); st.stop()

        r2 = r2_score(y, y_pred); r = np.sqrt(abs(r2))

        st.markdown("**Parámetros estimados (curve_fit):**")
        if not modo_presentacion: st.dataframe(params_df, use_container_width=True)

        st.markdown("**Métricas del ajuste:**")
        st.dataframe(pd.DataFrame({"R^2":[r2], "R ":[r]}), use_container_width=True)

        fig = px.scatter(x=x, y=y, labels={"x": Variable_x, "y": Variable_y},
                         opacity=0.6, title=f"{Modelo} — Dispersión y curva ajustada")
        fig.add_trace(go.Scatter(x=x_sorted, y=y_line, mode="lines", name="Ŷ (curva)", line=dict(width=2)))
        st.plotly_chart(fig, use_container_width=True)

        resid = y - y_pred
        fig_resid = px.scatter(x=y_pred, y=resid, labels={"x":"Ŷ", "y":"Residual"},
                               title="Residuos vs Predicción")
        fig_resid.add_hline(y=0, line_dash="dot")
        st.plotly_chart(fig_resid, use_container_width=True)

    except RuntimeError as e:
        st.error(f"No convergió el ajuste: {e}.")
    except Exception as e:
        st.error(f"Error durante el ajuste: {e}")

##########################################################################################
# Vista 4 — Regresión Logística
if View == "Regresión Logística":
    st.title("Regresión Logística")

    numeric_df = df.select_dtypes(include=['float', 'float64', 'int', 'int64'])
    Lista_num  = list(numeric_df.columns)

    dico_cols = []
    for col in df.columns:
        vals = df[col].dropna().unique()
        if len(vals) == 2:
            dico_cols.append(col)

    if len(dico_cols) == 0:
        st.warning("No se detectaron variables binarias en el dataset."); st.stop()

    Variable_y = st.sidebar.selectbox("Variable dependiente (Y, binaria)", options=dico_cols)
    Variables_x = st.sidebar.multiselect("Variables independientes (X, numéricas)", options=Lista_num)

    test_size = st.sidebar.slider("Tamaño de prueba", 0.1, 0.5, 0.30, 0.05)
    thr = st.sidebar.slider("Umbral de clasificación", 0.05, 0.95, 0.50, 0.01)

    if len(Variables_x) == 0:
        st.info("Selecciona al menos una variable independiente (X).")
    else:
        base = df[Variables_x + [Variable_y]].copy()
        vals = base[Variable_y].dropna().unique().tolist()
        if len(vals) != 2:
            st.error(f"La variable '{Variable_y}' debe tener exactamente 2 clases. Encontradas: {vals}")
            st.stop()

        mapping = {vals[0]: 0, vals[1]: 1}
        base['__y__'] = base[Variable_y].map(mapping)
        base = base.replace([np.inf, -np.inf], np.nan).dropna(subset=Variables_x + ['__y__'])
        if base['__y__'].nunique() < 2:
            st.error("Tras limpiar datos, solo queda una clase en Y."); st.stop()

        X = base[Variables_x].astype(float).to_numpy()
        y = base['__y__'].to_numpy(dtype=int)
        clases = vals

        st.sidebar.markdown("### Manejo de desbalance")
        imb_method = st.sidebar.selectbox("Método", ["Ninguno","class_weight='balanced'","SMOTE (over-sampling)","Under-sampling"])

        st.sidebar.markdown("### Estrategia de umbral")
        thr_mode = st.sidebar.selectbox("Seleccionar umbral por…", ["Manual", "F1 óptimo", "Minimizar costo", "Maximizar recall con precisión mínima"])
        prec_min = None; c_fp = None; c_fn = None
        if thr_mode == "Manual":
            thr = st.sidebar.slider("Umbral de clasificación", 0.01, 0.99, thr, 0.01)
        elif thr_mode == "Maximizar recall con precisión mínima":
            prec_min = st.sidebar.slider("Precisión mínima requerida", 0.1, 0.99, 0.6, 0.01)
        elif thr_mode == "Minimizar costo":
            c_fp = st.sidebar.number_input("Costo por FP", min_value=0, value=10000, step=1000)
            c_fn = st.sidebar.number_input("Costo por FN", min_value=0, value=80000, step=1000)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        escalar = StandardScaler()
        X_train_s = escalar.fit_transform(X_train)
        X_test_s  = escalar.transform(X_test)

        if imb_method == "SMOTE (over-sampling)":
            sm = SMOTE(random_state=42); X_train_s, y_train = sm.fit_resample(X_train_s, y_train)
        elif imb_method == "Under-sampling":
            rus = RandomUnderSampler(random_state=42); X_train_s, y_train = rus.fit_resample(X_train_s, y_train)

        if imb_method == "class_weight='balanced'":
            algoritmo = LogisticRegression(max_iter=1000, class_weight='balanced')
        else:
            algoritmo = LogisticRegression(max_iter=1000)
        algoritmo.fit(X_train_s, y_train)

        y_proba = algoritmo.predict_proba(X_test_s)[:, 1]

        def pick_threshold_by_f1(y_true, y_score):
            p, r, th = precision_recall_curve(y_true, y_score)
            f1 = 2 * (p*r) / np.clip(p+r, 1e-12, None); best_idx = np.nanargmax(f1[:-1])
            return th[best_idx], f1[best_idx], p[best_idx], r[best_idx]

        def pick_threshold_by_cost(y_true, y_score, c_fp, c_fn):
            ths = np.linspace(0.0, 1.0, 1001); best_th, best_cost = 0.5, np.inf
            for t in ths:
                y_pred = (y_score >= t).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                cost = fp * c_fp + fn * c_fn
                if cost < best_cost: best_cost, best_th = cost, t
            return best_th, best_cost

        def pick_threshold_by_recall_with_prec_min(y_true, y_score, prec_min=0.6):
            p, r, th = precision_recall_curve(y_true, y_score)
            valid = np.where(p[:-1] >= prec_min)[0]
            if len(valid) == 0: return 0.5, 0.0, 0.0
            best_idx = valid[np.argmax(r[valid])]; return th[best_idx], r[best_idx], p[best_idx]

        if thr_mode == "F1 óptimo":
            thr, best_f1, best_p, best_r = pick_threshold_by_f1(y_test, y_proba)
        elif thr_mode == "Minimizar costo":
            thr, best_cost = pick_threshold_by_cost(y_test, y_proba, c_fp, c_fn)
        elif thr_mode == "Maximizar recall con precisión mínima":
            thr, best_r, best_p = pick_threshold_by_recall_with_prec_min(y_test, y_proba, prec_min=prec_min)

        y_pred = (y_proba >= thr).astype(int)

        acc     = accuracy_score(y_test, y_pred)
        bacc    = balanced_accuracy_score(y_test, y_pred)
        prec_c0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
        prec_c1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec_c0  = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        rec_c1  = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_min  = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        auc     = roc_auc_score(y_test, y_proba)
        auprc   = average_precision_score(y_test, y_proba)

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

        prev = y_test.mean()
        if prec_c1 == 1.0 and rec_c1 < 0.15:
            st.warning("La precisión de la clase minoritaria es 1.0 pero el recall es muy bajo. Ajusta umbral o balanceo.")
        if acc > 0.9 and bacc < 0.65 and prev < 0.25:
            st.info("La exactitud es alta por el desbalance. Revisa balanced accuracy, AUPRC y F1 de la minoritaria.")

        coef = algoritmo.coef_[0]; intercepto = algoritmo.intercept_[0]
        coef_tab = pd.DataFrame({
            "Variable": ["Intercepto"] + Variables_x,
            "Coeficiente (log-odds)": [intercepto] + list(coef),
            "Odds Ratio (exp(coef))": [np.exp(intercepto)] + list(np.exp(coef))
        })
        if not modo_presentacion:
            st.subheader("Coeficientes del modelo")
            st.dataframe(coef_tab, use_container_width=True)

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
        fig_cm.update_layout(title=f"Matriz de confusión (umbral={thr:.2f})", annotations=ann, width=520, height=520)
        st.plotly_chart(fig_cm, use_container_width=False)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Aleatorio", line=dict(dash="dot")))
        fig_roc.update_layout(title="Curva ROC", xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig_roc, use_container_width=True)

        p, r, th = precision_recall_curve(y_test, y_proba)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=r, y=p, mode="lines", name=f"PR (AP={auprc:.3f})"))
        fig_pr.update_layout(title="Curva Precisión-Recall (clase 1)", xaxis_title="Recall", yaxis_title="Precisión")
        st.plotly_chart(fig_pr, use_container_width=True)

        fig_prob = px.strip(
            x=[labels_disp[i] for i in y_test], y=y_proba,
            labels={"x":"Clase real", "y":"Probabilidad P(Y=1)"},
            title="Distribución de probabilidades por clase real"
        )
        fig_prob.add_hline(y=thr, line_dash="dot", annotation_text=f"Umbral {thr:.2f}")
        st.plotly_chart(fig_prob, use_container_width=True)

        st.caption(f"Mapeo interno (solo para el modelo): {clases[0]} → 0, {clases[1]} → 1. Prevalencia clase 1 (test): {prev:.3f}")

##########################################################################################
# Vista 5 — COMPARAR PAÍSES (Nueva)
if View == "Comparar países":
    st.title("Comparación de países (Alemania · Valencia · Estocolmo · México)")
    st.caption("Misma métrica y visual por país, en una sola vista.")

    # Cargar todos los países
    dfs = {}
    listas_cat = {}
    for c in COUNTRY_FILES.keys():
        dfi, Li = load_country_df(c)
        dfs[c] = dfi
        listas_cat[c] = set(Li).intersection(set(dfi.columns))

    common_num, common_bin, common_cat = get_common_lists(dfs)

    # Sub-vistas
    subview = st.radio("Sub-vista", ["Extracción comparada", "Regresión logística comparada"], horizontal=True)

    if subview == "Extracción comparada":
        if len(common_cat)==0:
            st.error("No hay columnas categóricas en común en los 4 datasets.")
            st.stop()
        var_cat = st.selectbox("Variable categórica común", options=sorted(common_cat), index=sorted(common_cat).index("room_type") if "room_type" in common_cat else 0)

        # KPI's por país (fila completa)
        st.markdown("### KPI's por país")
        for c in COUNTRY_FILES.keys():
            with st.container():
                kpis_block(dfs[c], c)

        st.markdown("---")
        st.markdown("### Extracción (4× gráficas por país)")
        # Grilla 2x2 por país con (bar, pie, donut, área) + detalle (box/heatmap) + galería
        for c in COUNTRY_FILES.keys():
            st.subheader(f"{c}")
            tabla, fig_bar, fig_pie, fig_donut, fig_area, detail_fig = extraction_charts(dfs[c], var_cat)
            colA, colB = st.columns(2)
            with colA: st.plotly_chart(fig_bar, use_container_width=True)
            with colB: st.plotly_chart(fig_pie, use_container_width=True)
            colC, colD = st.columns(2)
            with colC: st.plotly_chart(fig_donut, use_container_width=True)
            with colD: st.plotly_chart(fig_area, use_container_width=True)

            st.plotly_chart(detail_fig, use_container_width=True)

            if not modo_presentacion:
                with st.expander(f"Tabla de frecuencias · {c}"):
                    st.dataframe(tabla.style.background_gradient(cmap='Reds'), use_container_width=True)

            with st.expander(f"Galería · {c}"):
                gallery_block(c)

            st.markdown("---")

    else:
        # Logística comparada
        if len(common_bin)==0:
            st.error("No hay variables binarias en común en los 4 datasets.")
            st.stop()
        if len(common_num)==0:
            st.error("No hay variables numéricas en común en los 4 datasets.")
            st.stop()

        st.markdown("### Parámetros comunes")
        y_col = st.selectbox("Variable Y (binaria, común)", options=common_bin)
        x_cols = st.multiselect("Variables X (numéricas, comunes)", options=common_num, default=[c for c in common_num if c not in [y_col]][:3])
        test_size = st.slider("Tamaño de prueba", 0.1, 0.5, 0.30, 0.05)

        colU, colV, colW = st.columns(3)
        with colU:
            imb_method = st.selectbox("Manejo de desbalance", ["Ninguno","class_weight='balanced'","SMOTE (over-sampling)","Under-sampling"])
        with colV:
            thr_mode = st.selectbox("Umbral por", ["Manual","F1 óptimo","Minimizar costo","Maximizar recall con precisión mínima"])
        with colW:
            thr_manual = st.slider("Umbral (si Manual)", 0.01, 0.99, 0.50, 0.01)

        colX, colY = st.columns(2)
        with colX:
            c_fp = st.number_input("Costo por FP (si Minimizar costo)", min_value=0, value=10000, step=1000)
        with colY:
            c_fn = st.number_input("Costo por FN (si Minimizar costo)", min_value=0, value=80000, step=1000)

        prec_min = st.slider("Precisión mínima (si Máx. recall)", 0.1, 0.99, 0.60, 0.01)

        if len(x_cols)==0:
            st.info("Selecciona al menos 1 X para correr comparación.")
            st.stop()

        # Ejecutar por país
        results = {}
        for c in COUNTRY_FILES.keys():
            res = run_logistic_block(
                dfs[c], y_col, x_cols,
                thr_mode=thr_mode,
                thr=thr_manual,
                c_fp=c_fp, c_fn=c_fn,
                prec_min=prec_min,
                test_size=test_size,
                imb_method=imb_method
            )
            if res is not None:
                results[c] = res

        if len(results)==0:
            st.error("No se pudo entrenar el modelo en ninguno de los países (revisa datos y clases).")
            st.stop()

        st.markdown("### Métricas comparadas")
        # Tabla apilada por país
        tabs = st.tabs(list(results.keys()))
        for tab, (c, res) in zip(tabs, results.items()):
            with tab:
                st.dataframe(res["metrics"], use_container_width=True)

        st.markdown("### Matrices de confusión por país")
        # Grilla 2x2
        countries = list(results.keys())
        rows = [countries[:2], countries[2:4]]
        for row in rows:
            cols = st.columns(len(row))
            for i, c in enumerate(row):
                with cols[i]:
                    st.markdown(f"**{c}**")
                    st.plotly_chart(results[c]["cm_fig"], use_container_width=True)

        with st.expander("Curvas ROC por país"):
            cols = st.columns(2)
            items = list(results.items())
            for i, (c, res) in enumerate(items):
                with cols[i%2]:
                    st.markdown(f"**{c}**"); st.plotly_chart(res["roc_fig"], use_container_width=True)

        with st.expander("Curvas Precisión-Recall por país"):
            cols = st.columns(2)
            items = list(results.items())
            for i, (c, res) in enumerate(items):
                with cols[i%2]:
                    st.markdown(f"**{c}**"); st.plotly_chart(res["pr_fig"], use_container_width=True)

        with st.expander("Distribución de probabilidades por país"):
            cols = st.columns(2)
            items = list(results.items())
            for i, (c, res) in enumerate(items):
                with cols[i%2]:
                    st.markdown(f"**{c}**"); st.plotly_chart(res["prob_fig"], use_container_width=True)

# FOOTER
st.markdown("---")
st.markdown("""
<div style="text-align:center; opacity:0.8; font-size:0.9rem;">
© Proyecto para Gestión de Proyectos — Dashboard creado por <b>Los Guaranies</b> con ayuda de IA y profe Freddy/Malu.  
<br> Construido con Streamlit, Plotly y Python.
</div>
""", unsafe_allow_html=True)


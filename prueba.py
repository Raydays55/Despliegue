# Corra Berlin_Final.py, por favor
# Se crea el archivo de la APP en el interprete principal (Phyton)
##########
# Importar librerías
import streamlit as st 
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import statsmodels.api as sm 
from statsmodels.formula.api import ols
from scipy import stats

# Personalización
# --- Page config (antes de cualquier st.*) ---
st.set_page_config(
    page_title="Airbnb — Berlín (Data App)",
    page_icon="assets/airbnb_favicon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS fino para look&feel Airbnb ---
AIRBNB_RED = "#FF5A5F"
AIRBNB_TEAL = "#00A699"
AIRBNB_ORANGE = "#FC642D"
AIRBNB_GRAY = "#767676"

st.markdown(
    f"""
    <style>
    /* Anchura del contenedor principal */
    .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; }}

    /* Titulares con acento */
    h1, h2, h3 {{
        letter-spacing: .2px;
    }}

    /* Botones */
    .stButton>button {{
        background:{AIRBNB_RED}; color:white; border-radius:12px; border:none;
        padding:.6rem 1rem; font-weight:600;
    }}
    .stButton>button:hover {{ opacity:.9 }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: #ffffff;
        border-right: 1px solid #eee;
    }}

    /* Cajas tipo "card" para KPIs */
    .air-card {{
        border: 1px solid #eee; border-radius:16px; padding:1rem; background:#fff;
        box-shadow: 0 1px 2px rgba(0,0,0,.03);
    }}

    /* Pie de página */
    .air-footer {{
        color:{AIRBNB_GRAY}; font-size:.9rem; margin-top:1.2rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# Definir instancia
@st.cache_resource
#####

# Crear función de carga de datos
def load_data():
    berlin = pd.read_csv('Berlin_86.csv')

    # Ajuste de variables
    df = berlin.drop(['Unnamed: 0','latitude', 'longitude'], axis=1)
    df['host_id'] = df['host_id'].astype(str)

    # Lista
    Lista =['host_is_superhost','host_identity_verified','host_response_time','host_response_rate','host_acceptance_rate','host_total_listings_count','host_verifications','room_type','property_type','price_cat']
    return df, Lista
####
# Carga de datos función 'load_data()'
df, Lista = load_data()

####
# Creación del dashboard
# Generar las páginas a utiizar en el diseño
####
# Generamos encabezados para barra lateral (sidebar)
st.sidebar.title('Berlín, Alemania')

# Widget 1: Selectbox
# Menú desplegable de opciones de las páginas seleccionadas
View = st.sidebar.selectbox(label= 'Tipo de análisis', options= ['Extracción de Características', 'Regresión Lineal', 'Regresión No Lineal', 'Regresión Logística'])

##########################################################################################
# CONTENIDO DE LA VISTA 1
if View == "Extracción de Características":

    Variable_Cat = st.sidebar.selectbox(label="Variable categórica a analizar", options=Lista)
    Tabla_frecuencias = df[Variable_Cat].value_counts().reset_index().head(10)
    Tabla_frecuencias.columns = ['categorias', 'frecuencia']

    st.title("Extracción de Características — Airbnb Berlín")
    #st.write(f"**Variable seleccionada:** {Variable_Cat}")
    st.caption('Se muestran máximo las 10 categorías con mas frecuencia')

    # Fila 1 — Barras Y Pastel
    Contenedor_A, Contenedor_B = st.columns(2)

    with Contenedor_A:
        st.subheader("Distribución por categoría (Bar Plot)")
        fig_bar = px.bar(
            Tabla_frecuencias,
            x='categorias',
            y='frecuencia',
            color='frecuencia',
            color_continuous_scale='Viridis',
            title=f"Frecuencia por categoría — {Variable_Cat}"
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

    with Contenedor_B:
        st.subheader("Proporción por categoría (Pie Chart)")
        fig_pie = px.pie(
            Tabla_frecuencias,
            names='categorias',
            values='frecuencia',
            color_discrete_sequence=px.colors.sequential.Tealgrn,
            title=f"Distribución porcentual — {Variable_Cat}"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Fila 2 — Anillo Y Área
    Contenedor_C, Contenedor_D = st.columns(2)

    with Contenedor_C:
        st.subheader("Visualización tipo dona")
        fig_donut = px.pie(
            Tabla_frecuencias,
            names='categorias',
            values='frecuencia',
            hole=0.5,
            color_discrete_sequence=px.colors.sequential.Mint,
            title=f"Gráfico de dona — {Variable_Cat}"
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with Contenedor_D:
        st.subheader("Tendencia acumulada (Área)")
        fig_area = px.area(
            Tabla_frecuencias.sort_values(by='frecuencia', ascending=False),
            x='categorias',
            y='frecuencia',
            title=f"Tendencia de frecuencia — {Variable_Cat}",
            color_discrete_sequence=['#5C8D89']
        )
        st.plotly_chart(fig_area, use_container_width=True)

    # Fila 3: Heatmap o Boxplot
    st.markdown("---")
    st.subheader("Análisis más profundo")

    if Variable_Cat in ['room_type', 'property_type', 'price_cat']:
        # Relacionar con variable numérica (precio)
        st.write("**Relación entre categorías y precio promedio (Boxplot):**")
        fig_box = px.box(
            df,
            x=Variable_Cat,
            y='price',
            color=Variable_Cat,
            title=f"Distribución de precios según {Variable_Cat}",
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        # Heatmap simple con frecuencia normalizada
        st.write("**Mapa de calor de proporciones (Heatmap):**")
        heat_df = pd.crosstab(index=df[Variable_Cat], columns='count', normalize='columns') * 100
        fig_heat = px.imshow(
            heat_df,
            color_continuous_scale='Viridis',
            title=f"Proporción por categoría — {Variable_Cat}",
            text_auto=".1f"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # Tabla de frecuencias
    st.markdown("---")
    st.subheader("Tabla de frecuencias")
    st.dataframe(Tabla_frecuencias.style.background_gradient(cmap='Blues'))

 ##########################################################################################
  # Contenido Vista 2
if View == "Regresión Lineal":
    st.title("Regresión Lineal")

    # Variables numéricas disponibles
    numeric_df = df.select_dtypes(include=['float', 'int']).copy()
    Lista_num = list(numeric_df.columns)

    # Lineal simple
    st.subheader("Regresión lineal simple")
    colL, colR = st.columns(2)
    with colL:
        Variable_y = st.selectbox("Variable dependiente (Y)", options=Lista_num, key="rl_y")
    with colR:
        Variable_x = st.selectbox("Variable independiente (X)", options=Lista_num, key="rl_x")

    # Ajuste
    X = numeric_df[[Variable_x]].values
    y = numeric_df[Variable_y].values
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
        "R: ": [coef_Correl_simple],
        "R^2: ": [coef_Deter_simple]
    })

    st.dataframe(coef_df_simple, use_container_width=True)

    # Graf: dispersión + recta y_pred
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
    fig_res = px.scatter(x=y_pred, y=resid, labels={"x":"Variable Independiente", "y":"Residual"},
                         title="Residuos vs Predicción (diagnóstico)")
    fig_res.add_hline(y=0, line_dash="dot")
    st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("---")

    # Lineal múltiple
    st.subheader("Regresión múltiple")
    col1, col2 = st.columns([1,2])
    with col1:
        Variable_y_M = st.selectbox("Variable dependiente (Y)", options=Lista_num, key="rlm_y")
    with col2:
        Variables_x_M = st.multiselect("Variables independientes (X)", options= Lista_num, key="rlm_xs")

    if len(Variables_x_M) >= 1:
        X_M = numeric_df[Variables_x_M].values
        y_M = numeric_df[Variable_y_M].values
        Model_M = LinearRegression()
        Model_M.fit(X_M, y_M)
        y_pred_M = Model_M.predict(X_M)

        # Métricas
        #Corroboramos cual es el coeficiente de Determinación de nuestro modelo
        coef_Deter_multiple= Model_M.score(X=X_M, y= y_M)
        #Corroboramos cual es el coeficiente de Correlación de nuestro modelo
        coef_Correl_multiple = np.sqrt(abs(coef_Deter_multiple))
        #r2M = r2_score(y_M, y_pred_M)
        n, p = X_M.shape

        # Coeficientes
        coef_tab = pd.DataFrame({
            "Variable": ["Intercepto"] + Variables_x_M,
            "Coeficiente": [Model_M.intercept_] + list(Model_M.coef_)
        })
        st.dataframe(coef_tab, use_container_width=True)

        met_tab = pd.DataFrame({'R^2': [coef_Deter_multiple], 'R ': [coef_Correl_multiple]})
        st.dataframe(met_tab, use_container_width=True)

        # Gráfica: Real vs Predicho 
        fig_pred = px.scatter(x=y_M, y=y_pred_M, labels={"x":"Y real ", "y": "Y predicciones"}, title="Comparación Y Real vs Y Predicciones") 
        fig_pred.add_trace(go.Scatter(x=[y_M.min(), y_M.max()], y=[y_M.min(), y_M.max()], mode="lines", name="Línea ideal", line=dict(dash="dot"))) 
        st.plotly_chart(fig_pred, use_container_width=True)

        # Mensaje
    else:
        st.info("Selecciona al menos 1 variable para el modelo múltiple.")


##########################################################################################
# Contenido Vista 3
if View == "Regresión No Lineal":
    st.title("Regresión No Lineal")

    # Variables numéricas
    numeric_df = df.select_dtypes(include=['float','int']).copy()
    Lista_num = list(numeric_df.columns)

    contA, contB = st.columns(2)
    with contA:
        Variable_y = st.selectbox("Variable objetivo (Y)", options=Lista_num, key="rnl_y_cf")
    with contB:
        Variable_x = st.selectbox("Variable independiente (X)", options=[c for c in Lista_num if c != Variable_y], key="rnl_x_cf")

    # Modelos disponibles
    modelos = ["Función cuadrática (a*x**2 + b*x + c)", "Función exponencial (a*np.exp(-b*x)+c)", "Función potencia (a*x**b)", "Función cúbica (a*x**3 + b*x**2 + c*x + d)"]
    Modelo = st.selectbox("Elige modelo no lineal", options=modelos, key="rnl_modelo_cf")

    # Datos
    x = numeric_df[Variable_x].to_numpy(dtype=float)
    y = numeric_df[Variable_y].to_numpy(dtype=float)
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]

    # Definiciones de funciones
    def func_cuad(x, a, b, c):
        return a*x**2 + b*x + c

    def func_cub(x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d

    def func_exp(x, a, b, c):
        # a*exp(bx)+c
        return a * np.exp(-b * x) + c

    def func_pot(x, a, b):
        # a*x^b
        return a * np.power(x, b)

    # Estimaciones iniciales p0 (robustas por defecto)
    # Se ajustan por modelo y por escala de los datos
    x_rng = np.ptp(x) if np.ptp(x) != 0 else 1.0
    y_rng = np.ptp(y) if np.ptp(y) != 0 else 1.0
    y_mean = np.nanmean(y)

    try:
        if Modelo == "Función cuadrática (a*x**2 + b*x + c)":
            pars, cov = curve_fit(func_cuad, x, y, maxfev=20000) # maxfev evita errores por iteraciones insuficientes
            y_pred = func_cuad(x, *pars)
            y_line = func_cuad(x_sorted, *pars)
            params_df = pd.DataFrame({"Parámetro": ["a", "b", "c"], "Valor": pars})

        elif Modelo == "Función cúbica (a*x**3 + b*x**2 + c*x + d)":
            pars, cov = curve_fit(func_cub, x, y, maxfev=30000)
            y_pred = func_cub(x, *pars)
            y_line = func_cub(x_sorted, *pars)
            params_df = pd.DataFrame({"Parámetro": ["a", "b", "c", "d"], "Valor": pars})

        elif Modelo == "Función exponencial (a*np.exp(-b*x)+c)":
            # Requiere y “razonables”. Filtramos si y es muy pequeña o negativa para evitar desbordes.
            mask = np.isfinite(y)
            if np.sum(mask) < 3:
                st.error("No hay suficientes datos válidos para ajustar el modelo exponencial.")
                st.stop()
            a0 = max(y) - min(y)
            b0 = 0.01 / x_rng
            c0 = min(y)
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
            b0 = 1.0
            a0 = np.exp(np.mean(np.log(y_pos) - b0*np.log(x_pos)))
            pars, cov = curve_fit(func_pot, x_pos, y_pos, maxfev=20000)
            # Predicciones en todo el rango (para evitar potencias de x<=0, usamos clip pequeño)
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
        st.dataframe(params_df, use_container_width=True)

        st.markdown("**Métricas del ajuste:**")
        st.dataframe(pd.DataFrame({"R^2":[r2], "R ":[r]}), use_container_width=True)

        # Gráfica: dispersión + curva predicha (ordenada por X)
        fig = px.scatter(x=x, y=y, labels={"x": Variable_x, "y": Variable_y},
                         opacity=0.6, title=f"{Modelo} — Dispersión y curva ajustada (curve_fit)")
        fig.add_trace(go.Scatter(x=x_sorted, y=y_line, mode="lines", name="Ŷ (curva)", line=dict(width=2)))
        st.plotly_chart(fig, use_container_width=True)

        # Residuos
        resid = y - y_pred
        fig_resid = px.scatter(x=y_pred, y=resid, labels={"x":"Ŷ", "y":"Residual"},
                               title="Residuos vs Predicción")
        fig_resid.add_hline(y=0, line_dash="dot")
        st.plotly_chart(fig_resid, use_container_width=True)

    except RuntimeError as e:
        st.error(f"No convergió el ajuste: {e}. Prueba con otra X/Y o revisa outliers.")
    except Exception as e:
        st.error(f"Error durante el ajuste: {e}")



##########################################################################################
# Contenido Vista 4
if View == "Regresión Logística":
    st.title("Regresión Logística")

    # 1) Listas base: Y binaria y X numéricas
    numeric_df = df.select_dtypes(include=['float', 'int'])
    Lista_num  = list(numeric_df.columns)

    # Detectar dicotómicas (exactamente 2 valores no nulos)
    # dico_cols = [c for c in df.columns if df[c].dropna().nunique() == 2]
    dico_cols = []
    for col in df.columns:
        vals = df[col].unique()
        if len(vals) == 2:
            dico_cols.append(col)

    # Decidí hacerlo por selectbox, no fue una buena idea
    #contA, contB = st.columns(2)
    #with contA:
    #    Variable_y = st.selectbox('Variable dependiente (Y, dicotómica)', options= dico_cols)
    #with contB:
    #    Variables_x = st.multiselect('Variables independientes (X)', options= Lista_num)

    # Sidebar
    Variable_y = st.sidebar.selectbox("Variable dependiente (Y, binaria)", options=dico_cols)
    Variables_x = st.sidebar.multiselect("Variables independientes (X, numéricas)", options=Lista_num)

    # Sliders
    test_size = st.sidebar.slider("Tamaño de prueba", 0.1, 0.5, 0.30, 0.05)
    thr = st.sidebar.slider("Umbral de clasificación", 0.05, 0.95, 0.50, 0.01)

    if len(Variables_x) == 0:
        st.info("Selecciona al menos una variable independiente (X).")
    else:
        # 2) Preparar X, y (sin modificar df original)
        X = df[Variables_x].astype(float).values
        y_raw = df[Variable_y]

        # Verificar binaria y mapear SOLO en memoria para el modelo
        clases = y_raw.dropna().unique().tolist()
        if len(clases) != 2:
            st.error(f"La variable '{Variable_y}' debe tener exactamente 2 clases. Encontradas: {clases}")
        else:
            mapping = {clases[0]: 0, clases[1]: 1}  # mantiene nombres originales para mostrar
            y = y_raw.map(mapping).values

            # 3) Split + escalado
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            escalar = StandardScaler()
            X_train_s = escalar.fit_transform(X_train)
            X_test_s  = escalar.transform(X_test)

            # 4) Modelo
            algoritmo = LogisticRegression()
            algoritmo.fit(X_train_s, y_train)

            # 5) Probabilidades y predicción con umbral
            y_proba = algoritmo.predict_proba(X_test_s)[:, 1]
            y_pred  = (y_proba >= thr).astype(int)

            # 6) Coeficientes
            acc    = accuracy_score(y_test, y_pred) # exactitud
            prec_c0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0) # precisión
            prec_c1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            rec_c0  = recall_score(y_test, y_pred, pos_label=0, zero_division=0) # sensibilidad
            rec_c1  = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            auc     = roc_auc_score(y_test, y_proba)

            met_tab = pd.DataFrame({
                "Métrica": ["Exactitud", f"Precision ({clases[0]})", f"Precision ({clases[1]})",
                            f"Sensibilidad ({clases[0]})", f"Sensibilidad ({clases[1]})", "ROC-AUC", "Umbral"],
                "Valor":   [acc,         prec_c0,                   prec_c1,
                            rec_c0,                  rec_c1,                  auc,      thr]
            })
            st.subheader("Métricas")
            st.dataframe(met_tab, use_container_width=True)

            # 7) Coeficientes y Odds Ratios con nombres de variables
            coef = algoritmo.coef_[0]
            intercepto = algoritmo.intercept_[0]
            coef_tab = pd.DataFrame({
                "Variable": ["Intercepto"] + Variables_x,
                "Coeficiente (log-odds)": [intercepto] + list(coef),
                "Odds Ratio (exp(coef))": [np.exp(intercepto)] + list(np.exp(coef))
            })
            st.subheader("Coeficientes del modelo")
            st.dataframe(coef_tab, use_container_width=True)

            # 8) Matriz de confusión con etiquetas originales
            matriz = confusion_matrix(y_test, y_pred, labels=[0, 1])
            labels_disp = [clases[0], clases[1]]  # orden consistente con mapping
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
            fig_cm.update_layout(
                title="Matriz de confusión",
                annotations=ann,
                width=520, height=520
            )
            st.plotly_chart(fig_cm, use_container_width=False)

            # 9) Curva ROC
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                         name=f"ROC (AUC={auc:.3f})"))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                         name="Aleatorio", line=dict(dash="dot")))
            fig_roc.update_layout(title="Curva ROC", xaxis_title="FPR", yaxis_title="TPR")
            st.plotly_chart(fig_roc, use_container_width=True)

            # 10) Probabilidades predichas por clase real + umbral
            fig_prob = px.strip(
                x=[labels_disp[i] for i in y_test], y=y_proba,
                labels={"x":"Clase real", "y":"Probabilidad P(Y=1)"},
                title="Distribución de probabilidades por clase real")

            fig_prob.add_hline(y=thr, line_dash="dot", annotation_text=f"Umbral {thr:.2f}")
            st.plotly_chart(fig_prob, use_container_width=True)

            # Nota informativa del mapeo (sin alterar df)
            st.caption(f"Mapeo interno (solo para el modelo, sin modificar el dataset): "
                       f"{clases[0]} → 0, {clases[1]} → 1")

##########################################################################################
# Personalización Dashboard




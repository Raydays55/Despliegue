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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import statsmodels.api as sm 
from statsmodels.formula.api import ols
from scipy import stats

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

    # FILA 1 — Barras Y Pastel
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

    # FILA 2 — Anillo Y Área
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

    # FILA 3: Heatmap o Boxplot
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
    st.subheader("Correlación lineal múltiple")
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

    colL, colR = st.columns(2)
    with colL:
        Variable_y = st.selectbox("Variable objetivo (Y)", options=Lista_num, key="rnl_y_cf")
    with colR:
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

    def func_pow(x, a, b):
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
            pars, cov = curve_fit(func_pow, x_pos, y_pos, maxfev=20000)
            # Predicciones en todo el rango (para evitar potencias de x<=0, usamos clip pequeño)
            x_safe = np.clip(x, 1e-12, None)
            x_sorted_safe = np.clip(x_sorted, 1e-12, None)
            y_pred = func_pow(x_safe, *pars)
            y_line = func_pow(x_sorted_safe, *pars)
            params_df = pd.DataFrame({"Parámetro": ["a", "b"], "Valor": pars})

        else:
            st.warning("Selecciona un modelo válido.")
            st.stop()

        # Métricas
        r2 = r2_score(y, y_pred)
        r = np.sqrt(abs(r2))

        # ---- Salidas
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
    st.title("Regresión Logística (binaria)")

    # 1) Detectar columnas binarias
    bin_cols = []
    for c in df.columns:
        vals = df[c].dropna().unique()
        if len(vals) == 2:
            bin_cols.append(c)

    if len(bin_cols) == 0:
        st.error("No se detectaron columnas binarias en el dataset.")
        st.stop()

    # 2) Mapeo automático a 0/1 si es object
    df_log = df.copy()
    bin_maps = {}
    for c in bin_cols:
        if df_log[c].dtype == 'object':
            # map a 0/1 con orden alfabético (puedes ajustar)
            uniques = sorted(df_log[c].dropna().unique().tolist())
            mapping = {uniques[0]:0, uniques[1]:1}
            df_log[c] = df_log[c].map(mapping)
            bin_maps[c] = mapping

    # X candidatas numéricas (evitar que Y se cuele)
    num_cols = list(df_log.select_dtypes(include=['float','int']).columns)

    colL, colR = st.columns(2)
    with colL:
        Variable_y = st.selectbox("Variable dependiente (binaria)", options=bin_cols, key="log_y")
    with colR:
        Variables_x = st.multiselect("Variables independientes (numéricas)", options=[c for c in num_cols if c != Variable_y], key="log_xs")

    if len(Variables_x) == 0:
        st.info("Selecciona al menos una X para entrenar el modelo.")
        st.stop()

    X = df_log[Variables_x].values
    y = df_log[Variable_y].values

    test_size = st.slider("Tamaño de prueba", 0.1, 0.5, 0.3, 0.05)
    thr = st.slider("Umbral de clasificación", 0.05, 0.95, 0.5, 0.01)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # Estandarización (tip: ayuda a convergencia)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    logit = LogisticRegression(max_iter=1000, solver="lbfgs")
    logit.fit(X_train_s, y_train)

    # Probabilidades y predicciones con umbral
    y_proba = logit.predict_proba(X_test_s)[:,1]
    y_pred = (y_proba >= thr).astype(int)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    met_df = pd.DataFrame({"Accuracy":[acc], "Precision":[prec], "Recall":[rec], "ROC-AUC":[auc], "Umbral":[thr]})
    st.dataframe(met_df, use_container_width=True)

    # Coeficientes y Odds Ratios
    coef = logit.coef_[0]
    intercepto = logit.intercept_[0]
    coef_tab = pd.DataFrame({
        "Variable": ["Intercepto"] + Variables_x,
        "Coeficiente (log-odds)": [intercepto] + list(coef),
        "Odds Ratio (exp(coef))": [np.exp(intercepto)] + list(np.exp(coef))
    })
    st.dataframe(coef_tab, use_container_width=True)

    # Matriz de confusión (heatmap)
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm, x=["Pred 0","Pred 1"], y=["Real 0","Real 1"],
        colorscale="Oranges", showscale=True, hoverongaps=False
    ))
    fig_cm.update_layout(title="Matriz de confusión", width=500, height=500)
    st.plotly_chart(fig_cm, use_container_width=False)

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Aleatorio", line=dict(dash="dot")))
    fig_roc.update_layout(title="Curva ROC", xaxis_title="FPR", yaxis_title="TPR")
    st.plotly_chart(fig_roc, use_container_width=True)

    # Dispersión de probabilidades predichas vs real
    fig_prob = px.strip(x=y_test, y=y_proba, labels={"x":"Clase real", "y":"Probabilidad(Y=1)"},
                        title="Distribución de probabilidades predichas por clase real", jitter=0.3)
    fig_prob.add_hline(y=thr, line_dash="dot", annotation_text=f"Umbral {thr:.2f}")
    st.plotly_chart(fig_prob, use_container_width=True)

    # Nota de mapping si hubo columnas binarias no numéricas
    if Variable_y in bin_maps:
        st.caption(f"Mapping aplicado a {Variable_y}: {bin_maps[Variable_y]}")



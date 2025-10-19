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
from sklearn.metrics import confusion_matrix
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
    df = pd.read_csv('Berlin_86.csv')
    # Relleno de nulos
    df = df.fillna(method= 'bfill')
    df = df.fillna(method= 'ffill')
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

    # FILA 2 — Dona Y Área
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
 ############################################################################


  # Contenido Vista 2
if View == "Regresión Lineal":
    # Variables numéricas disponibles
    numeric_df = df.select_dtypes(include=['float', 'int'])

    # Lista de col num
    Lista_num = list(numeric_df.columns)

    st.title("Regresión Lineal")

    # Simple
    st.subheader("Regresión Lineal Simple")
    colS1, colS2 = st.columns(2)
    with colS1:
        Variable_y = st.selectbox("Variable objetivo (Y)", options=Lista_num, key="rls_y")
    with colS2:
        Variable_x = st.selectbox("Variable independiente (X)", options=[c for c in Lista_num if c != Variable_y], key="rls_x")

    # Entrenamiento modelo simple
    X_s = numeric_df[[Variable_x]].values
    y_s = numeric_df[Variable_y].values
    model_s = LinearRegression()
    model_s.fit(X_s, y_s)
    y_pred_s = model_s.predict(X_s)

    # Métricas
    r2_s = r2_score(y_s, y_pred_s)
    mae_s = mean_absolute_error(y_s, y_pred_s)
    rmse_s = mean_squared_error(y_s, y_pred_s, squared=False)

    m = model_s.coef_[0]
    b = model_s.intercept_

    met_col1, met_col2, met_col3, met_col4, met_col5 = st.columns(5)
    met_col1.metric("R² (simple)", f"{r2_s:.3f}")
    met_col2.metric("MAE", f"{mae_s:.3f}")
    met_col3.metric("RMSE", f"{rmse_s:.3f}")
    met_col4.metric("Pendiente (β1)", f"{m:.3f}")
    met_col5.metric("Intercepto (β0)", f"{b:.3f}")

    # Figura: dispersión + línea de predicción
    fig_s = px.scatter(numeric_df, x=Variable_x, y=Variable_y, title="Lineal simple: datos vs. recta ajustada")
    # Línea ordenada por X para que no zigzaguee
    order_idx = np.argsort(X_s[:, 0])
    fig_s.add_trace(
        go.Scatter(
            x=X_s[order_idx, 0],
            y=y_pred_s[order_idx],
            mode="lines",
            name="Predicción",
        )
    )
    st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("---")

    # Múltiple
    st.subheader(" Regresión Lineal Múltiple")
    # Sugerimos por defecto las 2-4 más correlacionadas con Y
    corrs = numeric_df.corr(numeric_df[Variable_y]).abs().sort_values(ascending=False)
    sugeridas = [c for c in corrs.index if c != Variable_y][:4]

    Variables_xM = st.multiselect(
        "Variables independientes (X)",
        options=[c for c in Lista_num if c != Variable_y],
        default=sugeridas
    )

    if len(Variables_xM) >= 1:
        X_m = numeric_df[Variables_xM].values
        y_m = numeric_df[Variable_y].values

        model_m = LinearRegression()
        model_m.fit(X_m, y_m)
        y_pred_m = model_m.predict(X_m)

        r2_m = r2_score(y_m, y_pred_m)
        mae_m = mean_absolute_error(y_m, y_pred_m)
        rmse_m = mean_squared_error(y_m, y_pred_m, squared=False)

        met2_1, met2_2, met2_3 = st.columns(3)
        met2_1.metric("R² (múltiple)", f"{r2_m:.3f}")
        met2_2.metric("MAE", f"{mae_m:.3f}")
        met2_3.metric("RMSE", f"{rmse_m:.3f}")

        # Gráfico: y real vs y pred (parity plot) + línea 45°
        fig_parity = go.Figure()
        fig_parity.add_trace(go.Scatter(
            x=y_m, y=y_pred_m, mode="markers", name="Predicciones"
        ))
        min_v = float(np.nanmin([y_m.min(), y_pred_m.min()]))
        max_v = float(np.nanmax([y_m.max(), y_pred_m.max()]))
        fig_parity.add_trace(go.Scatter(
            x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="y = ŷ", line=dict(dash="dash")
        ))
        fig_parity.update_layout(title="Real vs. Predicho (Múltiple)", xaxis_title="y real", yaxis_title="y predicho")
        st.plotly_chart(fig_parity, use_container_width=True)

        # Importancias (magnitud de coeficientes estándar aprox. sin escalar)
        coefs = pd.Series(model_m.coef_, index=Variables_xM).sort_values(key=np.abs, ascending=False)
        st.caption("Coeficientes del modelo múltiple")
        st.bar_chart(coefs)
    else:
        st.info("Selecciona al menos una variable independiente para el modelo múltiple.")




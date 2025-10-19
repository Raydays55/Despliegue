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
from sklearn.linear_model import LinearRegression
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
 ############################################################################


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
    coef_Correl_simple = np.sqrt(coef_Deter_simple)

    # Coeficientes
    coef_df_simple = pd.DataFrame({
        "Variable": [Variable_x],
        "Coeficiente": [model.coef_[0]],
        "Intercepto": [model.intercept_],
        "R^2": [r2],
        "R: ": [coef_Correl_simple],
        "My R^2: ": [coef_Deter_simple]
    })

    st.dataframe(coef_df_simple, use_container_width=True)

    # Graf: dispersión + recta y_pred
    fig_scat = px.scatter(numeric_df, x=Variable_x, y=Variable_y, opacity=0.6, title="Dispersión y recta ajustada")
    # Línea predicha ordenando por X
    order_idx = np.argsort(X[:, 0])
    fig_scat.add_trace(go.Scatter(
        x=X[order_idx, 0], y=y_pred[order_idx],
        mode="lines", name="Predicción (Ŷ)"
    ))
    st.plotly_chart(fig_scat, use_container_width=True)

    # Residuales
    resid = y - y_pred
    fig_res = px.scatter(x=y_pred, y=resid, labels={"x":"Ŷ", "y":"Residual"},
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
        coef_Correl_multiple = np.sqrt(coef_Deter_multiple)
        r2M = r2_score(y_M, y_pred_M)
        n, p = X_M.shape

        # Coeficientes
        coef_tab = pd.DataFrame({
            "Variable": ["Intercepto"] + Variables_x_M,
            "Coeficiente": [Model_M.intercept_] + list(Model_M.coef_)
        })
        st.dataframe(coef_tab, use_container_width=True)

        met_tab = pd.DataFrame({"R^2":[r2M], 'My R^2': [coef_Deter_multiple], 'R ': [coef_Correl_multiple]})
        st.dataframe(met_tab, use_container_width=True)

        # Gráfica: Real vs Predicho
        fig_pred = px.scatter(x=y_M, y=y_pred_M, labels={"x":"Y real", "y":"Ŷ predicho"},
                              title="Comparación Real vs Predicho")
        fig_pred.add_trace(go.Scatter(x=[y_M.min(), y_M.max()], y=[y_M.min(), y_M.max()],
                                      mode="lines", name="Línea ideal", line=dict(dash="dot")))
        st.plotly_chart(fig_pred, use_container_width=True)

        # Residuales múltiple
        residM = y_M - y_pred_M
        fig_resM = px.scatter(x=y_pred_M, y=residM, labels={"x":"Ŷ", "y":"Residual"},
                              title="Residuos vs Predicción (múltiple)")
        fig_resM.add_hline(y=0, line_dash="dot")
        st.plotly_chart(fig_resM, use_container_width=True)
    else:
        st.info("Selecciona al menos 1 variable para el modelo múltiple.")


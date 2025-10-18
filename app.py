# Se crea el archivo de la APP en el interprete principal (Phyton)
##########
# Importar librerías
import streamlit as st 
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
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
    Lista = ['host_response_time', 'host_response_rate', 'room_type', 'property_type']
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
    Tabla_frecuencias = df[Variable_Cat].value_counts().reset_index()
    Tabla_frecuencias.columns = ['categorias', 'frecuencia']

    st.title("Extracción de Características — Airbnb Berlín")
    #st.write(f"**Variable seleccionada:** {Variable_Cat}")

    # FILA 1 — Barras Y Pastel
    ContA, ContB = st.columns(2)

    with ContA:
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

    with ContB:
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
    ContC, ContD = st.columns(2)

    with ContC:
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

    with ContD:
        st.subheader("Tendencia acumulada (Área)")
        fig_area = px.area(
            Tabla_frecuencias.sort_values(by='frecuencia', ascending=False),
            x='categorias',
            y='frecuencia',
            title=f"Tendencia de frecuencia — {Variable_Cat}",
            color_discrete_sequence=['#5C8D89']
        )
        st.plotly_chart(fig_area, use_container_width=True)

    # FILA 3 — BONUS: HEATMAP o BOXPLOT
    st.markdown("---")
    st.subheader("Análisis avanzado (Bonus Extra)")

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
    st.subheader("📋 Tabla de frecuencias")
    st.dataframe(Tabla_frecuencias.style.background_gradient(cmap='Blues'))
 ############################################################################






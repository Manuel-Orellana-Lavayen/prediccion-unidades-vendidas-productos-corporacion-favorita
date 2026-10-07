import pandas as pd
import polars as pl
import streamlit as st
from datetime import date
import base64
from datetime import datetime
import requests
import plotly.io as pio
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

df_productos = pl.scan_csv(BASE_DIR.parent/"API/Elementos/Recursos/PreparacionDatos/BaseConocimiento/productos.csv")

st.set_page_config(page_title= "Predictor de Unidades Vendidas- Manuel Orellana",layout= "wide")
st.title("Predictor de unidades vendidas",text_alignment= "center" )

diccionario_urls = {
    "Predicción Individual": ["http://127.0.0.1:8000/PredecirIndividual"],
    "Predicción Individual Recursiva": ["http://127.0.0.1:8000/PredecirRecursivoIndividual"],
    "Predicción Familia" : ["http://127.0.0.1:8000/PredecirFamilia"]
}

#===============================================================================================#
# Predicción individual por código de producto (método Recursivo y valor anterior)              #
#===============================================================================================#
st.subheader("Predicción Individual",text_alignment= "center" )
# Solicitando Valores
with st.container(border=True):
    fecha = st.slider(label = "Fecha a predecir", value = date(2017, 1, 3), min_value= date(2017, 1, 2), max_value=date(2017, 8, 15), format="YYYY-MM-DD", key = "barra 1")
    fecha = datetime.strftime(fecha, "%Y-%m-%d")
    numero_local = st.selectbox(label= "Número de local", options= range(1, 53), key = "num_tienda 1")

    opciones_familia = df_productos.select(["familia_producto"]).unique().collect()
    familia_producto = st.selectbox(label = "Familia Producto", options= opciones_familia, on_change= st.rerun, key= "selector_familia 1")

    opciones_codigo_producto = df_productos.filter(pl.col("familia_producto") == familia_producto).select(["numero_articulo"]).unique().collect().to_series().to_list()
    codigo_producto = st.selectbox(label= "Código del producto", options= opciones_codigo_producto, key= "selector_producto" )

    en_promocion = st.radio(label="Producto en Promoción", options=["Si", "No"])

    metodo_prediccion = st.radio(label="Método de Predicción", options=["Valor Anterior", "Recursivo"])
    enviar = st.button(label= "Predecir", key= "Predecir 1")

# Realizando Predicción con base en valores
if enviar:
    if en_promocion == "Si":
        en_promocion = 1
    else:
        en_promocion = 0

    if metodo_prediccion == "Valor Anterior":
        respuesta = requests.post(
                                    url= diccionario_urls["Predicción Individual"][0], #La url de la función
                                    json={"numero_tienda": numero_local, "numero_articulo": codigo_producto, "fecha": fecha, "en_promocion": en_promocion} #Lo que se le envia a la función
                                 )
    else:
        respuesta = requests.post(
                                    url=diccionario_urls["Predicción Individual Recursiva"][0],  # La url de la función
                                    json={"numero_tienda": numero_local, "numero_articulo": codigo_producto, "fecha": fecha,"en_promocion": en_promocion}  # Lo que se le envia a la función
                                 )

    if respuesta.status_code == 200:
        datos = respuesta.json()

        prediccion= datos["Predicción"]
        unidades_vendidas_real = datos["Unidadesreales"]
        if metodo_prediccion == "Valor Anterior":
            texto_base64 = datos["Figurabase64"]
            grafico_explicativo = base64.b64decode(texto_base64)


        with st.container(border= True):
            st.metric(
                label="Predicción",
                value=prediccion,
            )
            st.metric(
                label="Unidades Vendidas reales",
                value=unidades_vendidas_real,
            )
            if metodo_prediccion == "Valor Anterior":
                st.subheader("¿Por qué el modelo tomó esta decisión?")
                st.image(grafico_explicativo)
    else:
        st.error("Error al comunicarse con la API")
        st.error(f"Error HTTP: {respuesta.status_code}")
        st.write(respuesta.text)

#===============================================================================================#
# Predicción por familia de productos (método valor anterior)                                   #
#===============================================================================================#
st.subheader("Predicción por Familia de productos", text_alignment="center")

# Solicitando Valores
with st.container(border=True):
    fecha = st.slider(label="Fecha a predecir", value=date(2017, 1, 3), min_value=date(2017, 1, 2),
                      max_value=date(2017, 8, 15), format="YYYY-MM-DD", key= "barra 2")
    fecha = datetime.strftime(fecha, "%Y-%m-%d")

    numero_local = st.selectbox(label="Número de local", options=range(1, 53), key = "num_tienda 2")

    opciones_familia = df_productos.select(["familia_producto"]).unique().collect()
    familia_producto = st.selectbox(label="Familia Producto", options=opciones_familia, on_change=st.rerun,
                                    key="selector_familia 2")

    enviar = st.button(label="Predecir", key= "Predecir 2")

# Realizando Predicción con base en valores
if enviar:
    respuesta = requests.post(
        url=diccionario_urls["Predicción Familia"][0],  # La url de la función
        json={"numero_tienda": numero_local, "nombre_familia": familia_producto, "fecha": fecha}  # Lo que se le envia a la función
    )

    if respuesta.status_code == 200:
        datos = respuesta.json()

        prediccion = datos["Predicción"]
        prediccion = pd.DataFrame(prediccion).sort_values(by= "prediccion", ascending= False).set_index("numero_articulo")
        grafico = datos["Gráfico Dispersion"]
        grafico = pio.from_json(grafico)
        texto_base64 = datos["Figurabase64"]
        grafico_explicativo = base64.b64decode(texto_base64)

        st.write("Grafico de dispersión de unidades vendidas por producto")
        st.write("(El gráfico es interactivo, puede pulsar la leyenda para visualizar cualquiera de los dos valores)")
        st.plotly_chart(grafico)
        st.write("Predicciones de unidades vendidas por producto")
        st.dataframe(prediccion)
        st.subheader("Factores globales que más influyen en las ventas")
        st.image(grafico_explicativo)

    else:
        st.error("Error al comunicarse con la API")
        st.error(f"Error HTTP: {respuesta.status_code}")
        st.write(respuesta.text)

# Autor al pie de página
st.caption("2026 - Desarrollado por: Manuel Elias Orellana Lavayen")
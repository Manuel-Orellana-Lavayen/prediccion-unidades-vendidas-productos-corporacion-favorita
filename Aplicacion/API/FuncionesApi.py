# Creado Por Manuel Elias Orellana Lavayen - 2026
"""
    Funciónes que Predicen

    Este módulo contiene 3 funciones que tienen como objetivo realizar los 3 tipos de predicciones que ofrece la
    aplicación, además de 1 función para generar un gráfico para la función de predicción en grupo de familia.

    - prediccion_individual
    - prediccion_recursiva_individual
    - prediccion_grupo_familia
    - grafico
"""

import polars as pl
from datetime import datetime
import pandas as pd
from collections import deque
import statistics
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import shap
import plotly.express as px
from API.Elementos.Recursos.EstilosGraficosPlotly.EstilosPlotly import estilos_plotly
from API.Elementos.Recursos.ModelosRegresion.Modelos import lista_modelos_cluster
from API.Elementos.Recursos.PreparacionDatos.FuncionesPreparacionDatos import obtener_datos_tienda, obtener_datos_fecha, obtener_datos_articulo, df_unidades_vendidas_log, mapeo_categoricas, df_productos, dict_numero_articulo, dict_familia_producto, dict_clase_producto

plt.rcParams["text.color"] = "white"  # Color de los textos generales
plt.rcParams["axes.labelcolor"] = "white"  # Color de las etiquetas de los ejes
plt.rcParams["xtick.color"] = "white"  # Color de los números del eje X
plt.rcParams["ytick.color"] = "white"  # Color de los nombres de las variables (Eje Y)

def prediccion_individual(numero_tienda: int, numero_articulo: int, fecha: str, en_promocion: int):
    """Realiza una predicción para un día específico, con un producto específico, con un local específico y si el estado
    del producto está en promoción o no está en promoción. lo hace basándose en el valor anterior real.

        Args:
           numero_tienda: int
           numero_articulo: int
           fecha: str
           en_promocion: int

        Returns:
            prediccion -> Predicción del modelo
            unidades_vendidas_real -> Valor real para comparar con la prediccion del modelo
            grafico_base64 -> Gráfico SHAP para ver qué variables fueron más relevantes en la predicción
        """
    es_feriado_nacional, es_feriado_local, es_feriado_regional, es_evento, mes_seno, mes_coseno, semana_seno, semana_coseno, dia_semana_seno, dia_semana_coseno, petroleo_promedio_3meses, es_quincena = obtener_datos_fecha(fecha)
    ciudad, provincia_estado, tipo_tienda, cluster = obtener_datos_tienda(numero_tienda)
    numero_articulo, familia_producto, clase_producto, perecedero = obtener_datos_articulo(numero_articulo)
    fecha = datetime.strptime(fecha,"%Y-%m-%d" )

    unidades_vendidas_ayer = (df_unidades_vendidas_log
                              .filter((pl.col("fecha") == fecha) & (pl.col("numero_articulo") == numero_articulo) & (pl.col("numero_tienda") == numero_tienda))
                              .select(["unidades_vendidas_ayer"])).collect(engine=  "streaming").item()


    unidades_vendidas_promedio_semanal = (df_unidades_vendidas_log
                                          .filter((pl.col("fecha") == fecha) & (pl.col("numero_articulo") == numero_articulo) & (pl.col("numero_tienda") == numero_tienda))
                                          .select(["unidades_vendidas_promedio_semanal"])).collect(engine=  "streaming").item()

    unidades_vendidas_real = (df_unidades_vendidas_log
                                          .filter((pl.col("fecha") == fecha) & (pl.col("numero_articulo") == numero_articulo) & (pl.col("numero_tienda") == numero_tienda))
                                          .select(["unidades_vendidas"])).collect(engine="streaming").item()


    familia_producto = mapeo_categoricas(familia_producto, "familia_producto")
    clase_producto = mapeo_categoricas(clase_producto, "clase_producto")
    numero_articulo = mapeo_categoricas(numero_articulo, "numero_articulo")

    ciudad = mapeo_categoricas(ciudad, "ciudad")
    provincia_estado = mapeo_categoricas(provincia_estado, "provincia_estado")
    tipo_tienda = mapeo_categoricas(tipo_tienda, "tipo_tienda")

    df_prediccion = pd.DataFrame(
        {
            "numero_tienda": numero_tienda ,
            "clase_producto": clase_producto,
            "familia_producto": familia_producto,
            "ciudad": ciudad,
            "provincia_estado": provincia_estado,
            "tipo_tienda": tipo_tienda,
            "numero_articulo": numero_articulo,
            "mes_seno": mes_seno,
            "mes_coseno": mes_coseno,
            "dia_semana_seno": dia_semana_seno,
            "dia_semana_coseno": dia_semana_coseno,
            "semana_seno": semana_seno,
            "semana_coseno": semana_coseno,
            "en_promocion": en_promocion,
            "perecedero": perecedero,
            "es_feriado_nacional": es_feriado_nacional,
            "es_feriado_local": es_feriado_local,
            "es_feriado_regional": es_feriado_regional,
            "es_evento": es_evento,
            "unidades_vendidas_ayer": unidades_vendidas_ayer,
            "unidades_vendidas_promedio_semanal": unidades_vendidas_promedio_semanal,
            "es_quincena": es_quincena,
            "petroleo_promedio_3meses": petroleo_promedio_3meses
        },
        index= [0])
    modelo = lista_modelos_cluster[cluster - 1]

    prediccion = modelo.predict(X=df_prediccion)
    prediccion = float(prediccion[0])

    # Obteniendo Grafico Shap
    explicador = shap.TreeExplainer(modelo)
    shap_valores = explicador(df_prediccion)

    plt.figure(figsize=(7, 5), facecolor="none")
    shap.waterfall_plot(shap_valores[0], max_display=20, show=False)

    fig = plt.gcf() # Atrapando grafico con matplotlib

    for ax in fig.axes:
        ax.set_facecolor("none")  # Asegurar fondo transparente interno
        # Cambiamos las líneas de los bordes del gráfico a color blanco/gris claro
        for spine in ax.spines.values():
            spine.set_color("#555555")

    # Guardar los píxeles en la memoria RAM (BytesIO)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close(fig)

    # Transformar a texto Base64 para meterlo al JSON de FastAPI
    grafico_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return prediccion, unidades_vendidas_real,  grafico_base64

def prediccion_recursiva_individual(numero_tienda: int, numero_articulo: int, fecha: str, en_promocion: int):
    """Realiza una predicción para un día específico, con un producto específico, con un local específico y si el estado
        del producto está en promoción o no está en promoción. Lo hace de manera recursiva para obtener el valor anterior

            Args:
               numero_tienda: int
               numero_articulo: int
               fecha: str
               en_promocion: int

            Returns:
                prediccion -> Predicción del modelo
                unidades_vendidas_real -> Valor real para comparar con la prediccion del modelo
            """
    ciudad, provincia_estado, tipo_tienda, cluster = obtener_datos_tienda(numero_tienda)
    numero_articulo, familia_producto, clase_producto, perecedero = obtener_datos_articulo(numero_articulo)

    unidades_vendidas_ayer_primera = (df_unidades_vendidas_log
                              .filter((pl.col("fecha") == datetime.strptime("2017-01-02","%Y-%m-%d" )) & (pl.col("numero_articulo") == numero_articulo) & (pl.col("numero_tienda") == numero_tienda))
                              .select(["unidades_vendidas_ayer"])).collect(engine="streaming").item()

    unidades_vendidas_promedio_semanal_primera = (df_unidades_vendidas_log
                                          .filter((pl.col("fecha") == datetime.strptime("2017-01-02","%Y-%m-%d" )) & (pl.col("numero_articulo") == numero_articulo) & (pl.col("numero_tienda") == numero_tienda))
                                          .select(["unidades_vendidas_promedio_semanal"])).collect(engine="streaming").item()

    unidades_vendidas_real = (df_unidades_vendidas_log
                                          .filter((pl.col("fecha") == datetime.strptime(fecha,"%Y-%m-%d" )) & (pl.col("numero_articulo") == numero_articulo) & (pl.col("numero_tienda") == numero_tienda))
                                          .select(["unidades_vendidas"])).collect(engine="streaming").item()

    familia_producto = mapeo_categoricas(familia_producto, "familia_producto")
    clase_producto = mapeo_categoricas(clase_producto, "clase_producto")
    numero_articulo = mapeo_categoricas(numero_articulo, "numero_articulo")

    ciudad = mapeo_categoricas(ciudad, "ciudad")
    provincia_estado = mapeo_categoricas(provincia_estado, "provincia_estado")
    tipo_tienda = mapeo_categoricas(tipo_tienda, "tipo_tienda")

    modelo = lista_modelos_cluster[cluster-1]

    fechas = pd.date_range(start="2017-01-02", end= fecha)
    prediccion_actual = None
    cola_predicciones = deque()

    for fecha in fechas:
        fecha = datetime.strftime(fecha, "%Y-%m-%d")
        es_feriado_nacional, es_feriado_local, es_feriado_regional, es_evento, mes_seno, mes_coseno, semana_seno, semana_coseno, dia_semana_seno, dia_semana_coseno, petroleo_promedio_3meses, es_quincena = obtener_datos_fecha(fecha)
        if len(cola_predicciones) == 8:
            cola_predicciones.popleft()

        if fecha == "2017-01-02":
            unidades_vendidas_ayer = unidades_vendidas_ayer_primera
            unidades_vendidas_promedio_semanal = unidades_vendidas_promedio_semanal_primera
        else:
            unidades_vendidas_ayer = prediccion_actual
            unidades_vendidas_promedio_semanal = statistics.fmean(cola_predicciones)

        df_prediccion = pd.DataFrame(
            {
                "numero_tienda": numero_tienda,
                "clase_producto": clase_producto,
                "familia_producto": familia_producto,
                "ciudad": ciudad,
                "provincia_estado": provincia_estado,
                "tipo_tienda": tipo_tienda,
                "numero_articulo": numero_articulo,
                "mes_seno": mes_seno,
                "mes_coseno": mes_coseno,
                "dia_semana_seno": dia_semana_seno,
                "dia_semana_coseno": dia_semana_coseno,
                "semana_seno": semana_seno,
                "semana_coseno": semana_coseno,
                "en_promocion": en_promocion,
                "perecedero": perecedero,
                "es_feriado_nacional": es_feriado_nacional,
                "es_feriado_local": es_feriado_local,
                "es_feriado_regional": es_feriado_regional,
                "es_evento": es_evento,
                "unidades_vendidas_ayer": unidades_vendidas_ayer,
                "unidades_vendidas_promedio_semanal": unidades_vendidas_promedio_semanal,
                "es_quincena": es_quincena,
                "petroleo_promedio_3meses": petroleo_promedio_3meses
            },
            index=[0])

        prediccion = modelo.predict(X= df_prediccion)
        prediccion = prediccion[0]
        cola_predicciones.append(prediccion)
        prediccion_actual = prediccion

    prediccion_actual = float(prediccion_actual)
    return prediccion_actual, unidades_vendidas_real

def prediccion_grupo_familia(numero_tienda: int, nombre_familia: str, fecha: str):
    """Realiza una predicción para un día específico, con una familia específica, con un local específico.
        Lo hace basándose en el valor anterior real.

            Args:
               numero_tienda: int
               nombre_familia: str
               fecha: str

            Returns:
                prediccion -> Tabla de Predicciones del modelo, donde se encuentra el valor real y el predicho
                grafico_base64 -> Gráfico SHAP para ver qué variables fueron más relevantes en la predicción
    """
    en_promocion = 0 # Consideramos las predicciones con todos los productos sim promociones
    es_feriado_nacional, es_feriado_local, es_feriado_regional, es_evento, mes_seno, mes_coseno, semana_seno, semana_coseno, dia_semana_seno, dia_semana_coseno, petroleo_promedio_3meses, es_quincena = obtener_datos_fecha(fecha)
    ciudad, provincia_estado, tipo_tienda, cluster = obtener_datos_tienda(numero_tienda)
    articulos = df_productos[df_productos["familia_producto"] == nombre_familia]
    articulos = articulos.sort_values(by= "numero_articulo")

    fecha = datetime.strptime(fecha,"%Y-%m-%d" )

    unidades_vendidas_ayer = (df_unidades_vendidas_log
                              .filter((pl.col("fecha") == fecha) & (pl.col("numero_articulo").is_in(articulos["numero_articulo"].unique())) & (pl.col("numero_tienda") == numero_tienda))
                              .select(["unidades_vendidas_ayer", "numero_articulo"])).collect(engine="streaming").to_pandas()

    unidades_vendidas_ayer = unidades_vendidas_ayer.sort_values(by= "numero_articulo")

    unidades_vendidas_promedio_semanal = (df_unidades_vendidas_log
                                          .filter((pl.col("fecha") == fecha) & (pl.col("numero_articulo").is_in(articulos["numero_articulo"].unique())) & (pl.col("numero_tienda") == numero_tienda))
                                          .select(["unidades_vendidas_promedio_semanal", "numero_articulo"])).collect(engine="streaming").to_pandas()

    unidades_vendidas_reales = (df_unidades_vendidas_log
                                          .filter((pl.col("fecha") == fecha) & (pl.col("numero_articulo").is_in(articulos["numero_articulo"].unique())) & (pl.col("numero_tienda") == numero_tienda))
                                          .select(["unidades_vendidas", "numero_articulo"])).collect(engine="streaming").to_pandas()

    unidades_vendidas_promedio_semanal = unidades_vendidas_promedio_semanal.sort_values(by="numero_articulo")

    articulos = pd.merge(
        articulos, unidades_vendidas_ayer,
        on="numero_articulo",
        how="inner"
    )

    articulos = pd.merge(
        articulos, unidades_vendidas_promedio_semanal,
        on="numero_articulo",
        how="inner"
    )

    articulos = pd.merge(
        articulos, unidades_vendidas_reales,
        on="numero_articulo",
        how="inner"
    )

    map_nombre_familia = dict_familia_producto[nombre_familia]

    articulos["numero_articulo_mapeado"] = articulos["numero_articulo"].astype(str)
    articulos["numero_articulo_mapeado"] = articulos["numero_articulo_mapeado"].map(dict_numero_articulo)

    articulos["clase_producto"] = articulos["clase_producto"].astype(str)
    articulos["clase_producto"] = articulos["clase_producto"].map(dict_clase_producto)

    articulos = articulos.drop(columns= ["familia_producto"])

    ciudad = mapeo_categoricas(ciudad, "ciudad")
    provincia_estado = mapeo_categoricas(provincia_estado, "provincia_estado")
    tipo_tienda = mapeo_categoricas(tipo_tienda, "tipo_tienda")

    df_prediccion = pd.DataFrame(
        {
            "numero_tienda": numero_tienda,
            "clase_producto": articulos["clase_producto"].to_numpy(),
            "familia_producto": map_nombre_familia,
            "ciudad": ciudad,
            "provincia_estado": provincia_estado,
            "tipo_tienda": tipo_tienda,
            "numero_articulo": articulos["numero_articulo_mapeado"].to_numpy(),
            "mes_seno": mes_seno,
            "mes_coseno": mes_coseno,
            "dia_semana_seno": dia_semana_seno,
            "dia_semana_coseno": dia_semana_coseno,
            "semana_seno": semana_seno,
            "semana_coseno": semana_coseno,
            "en_promocion": en_promocion,
            "perecedero": articulos["perecedero"].to_numpy(),
            "es_feriado_nacional": es_feriado_nacional,
            "es_feriado_local": es_feriado_local,
            "es_feriado_regional": es_feriado_regional,
            "es_evento": es_evento,
            "unidades_vendidas_ayer": articulos["unidades_vendidas_ayer"].to_numpy(),
            "unidades_vendidas_promedio_semanal": articulos["unidades_vendidas_promedio_semanal"].to_numpy(),
            "es_quincena": es_quincena,
            "petroleo_promedio_3meses": petroleo_promedio_3meses
        })
    modelo = lista_modelos_cluster[cluster - 1]

    prediccion = modelo.predict(X=df_prediccion)

    articulos["prediccion"] = prediccion
    articulos= articulos[["numero_articulo", "prediccion", "unidades_vendidas"]]

    # Obteniendo Grafico Shap
    explicador = shap.TreeExplainer(modelo)
    shap_valores = explicador(df_prediccion)

    plt.figure(figsize=(2, 1), facecolor="none")
    shap.summary_plot(shap_valores, max_display=10, show=False)

    fig = plt.gcf()  # Atrapando grafico con matplotlib

    for ax in fig.axes:
        ax.set_facecolor("none")  # Asegurar fondo transparente interno
        # Cambiamos las líneas de los bordes del gráfico a color blanco/gris claro
        for spine in ax.spines.values():
            spine.set_color("#555555")


    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close(fig)

    grafico_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return articulos.to_dict(), grafico_base64



def grafico (predicciones: dict):
    """Hace un gráfico de dispersion con las unidades vendidas reales y unidades vendidas predichas con base en la tabla
       de predicciones por grupo de familia de productos.

            Args:
               predicciones: dict

            Returns:
                grafico_dispersion
    """
    predicciones = pd.DataFrame(predicciones)
    predicciones = predicciones.sort_values(by = "prediccion")
    predicciones["numero_articulo"] = predicciones["numero_articulo"].astype(str)

    df_melted = predicciones.melt(
        id_vars=   ["numero_articulo"],
        value_vars=["prediccion",  "unidades_vendidas"],
        var_name="Tipo de Venta",
        value_name="Unidades Vendidas"
    )

    # Renombrar las etiquetas
    df_melted["Tipo de Venta"] = df_melted["Tipo de Venta"].map({
        "prediccion": "Venta Predicha",
        "unidades_vendidas": "Venta Real"
    })

    grafico_dispersion = px.scatter(df_melted, x= "numero_articulo", y= "Unidades Vendidas", color="Tipo de Venta", labels={"numero_articulo": "Número Artículo", "y": "Unidades Vendidas"}, color_discrete_map={"Real": "#1f77b4", "Predicha": "#ff7f0e"})
    grafico_dispersion = estilos_plotly(figura= grafico_dispersion, titulo= "Predicciones del ventas" , color_titulo= "#FFFFFF", titulo_leyenda= "Metricas", color_texto_leyenda="#FFFFFF", fondo_transparente= True, leyenda= True, json= True)

    return grafico_dispersion
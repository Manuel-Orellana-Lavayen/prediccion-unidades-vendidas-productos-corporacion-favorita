# Creado Por Manuel Elias Orellana Lavayen - 2026
"""
    Funciones de Preparación de Datos

    Este módulo contiene las funciones utilizadas para preparar los datos para ser utilizados al momento de realizar alguna predicción

    Se proporcionan las siguientes funciones:

    - obtener_datos_fecha -> Busca obtener toda la información posible con base en la fecha
    - obtener_datos_tienda -> Busca obtener toda la información posible con base en el número de tienda
    - obtener_datos_articulo -> Busca obtener toda la información posible con base en el número del artículo
    - mapeo_categoricas -> recibe una variable y la mapea según el dato con el que el modelo haya sido entrenado
"""

import pandas as pd
import polars as pl
import numpy as np
import math
import json
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

df_eventos = pd.read_csv(BASE_DIR / "BaseConocimiento/eventos.csv")
df_holidays_local = pd.read_csv(BASE_DIR / "BaseConocimiento/holidays_local.csv")
df_holidays_nacional = pd.read_csv(BASE_DIR / "BaseConocimiento/holidays_nacional.csv")
df_holidays_regional = pd.read_csv(BASE_DIR / "BaseConocimiento/holidays_regional.csv")

df_petroleo_promedio_3meses = pd.read_csv(BASE_DIR / "BaseConocimiento/petroleo_promedio_3meses.csv")

df_tiendas = pd.read_csv(BASE_DIR / "BaseConocimiento/tiendas.csv")

df_productos = pd.read_csv(BASE_DIR / "BaseConocimiento/productos.csv")

df_unidades_vendidas_log = pl.scan_parquet(BASE_DIR / "BaseConocimiento/testeo_2017.parquet")

with open(BASE_DIR / "CodificadoresCategoricas/ciudad.json", "r", encoding="utf-8") as f:
    dict_ciudad = json.load(f)

with open(BASE_DIR / "CodificadoresCategoricas/clase_producto.json", "r", encoding="utf-8") as f:
    dict_clase_producto = json.load(f)

with open(BASE_DIR / "CodificadoresCategoricas/familia_producto.json", "r", encoding="utf-8") as f:
    dict_familia_producto = json.load(f)

with open(BASE_DIR / "CodificadoresCategoricas/numero_articulo.json", "r", encoding="utf-8") as f:
    dict_numero_articulo = json.load(f)

with open(BASE_DIR / "CodificadoresCategoricas/provincia_estado.json", "r", encoding="utf-8") as f:
    dict_provincia_estado = json.load(f)

with open(BASE_DIR / "CodificadoresCategoricas/tipo_tienda.json", "r", encoding="utf-8") as f:
    dict_tipo_tienda = json.load(f)


DOS_PI = 2 * np.pi

def obtener_datos_fecha (fecha):
    """Crea diferentes variables con base en una fecha específica.

        Args:
            fecha : str

        Returns:
            es_feriado_nacional: bool
            es_feriado_local: bool
            es_feriado_regional: bool
            es_evento: bool
            mes_seno: float
            mes_coseno: float
            semana_seno : float
            semana_coseno : float
            dia_semana_seno : float
            dia_semana_coseno : float
            petroleo_promedio_3meses : float
            es_quincena : bool
        """
    fecha = datetime.strptime(fecha,"%Y-%m-%d")

    mes = fecha.month
    dia_semana = fecha.weekday()
    semana = fecha.isocalendar().week
    dia = fecha.day
    dia_y_mes = fecha.strftime("%m-%d")

    mes = (mes * DOS_PI)/12
    semana = (semana * DOS_PI) / 53
    dia_semana = (dia_semana * DOS_PI) / 7

    es_feriado_nacional = 1 if dia_y_mes in df_holidays_nacional["dia-mes"].unique()  else 0
    es_feriado_local = 1 if dia_y_mes in df_holidays_local["dia-mes"].unique()  else 0
    es_feriado_regional = 1 if dia_y_mes in df_holidays_regional["dia-mes"].unique() else 0
    es_evento = 1 if datetime.strftime(fecha, "%Y-%m-%d") in df_eventos["fecha"].unique() else 0

    mes_coseno = math.cos(mes)
    mes_seno = math.sin(mes)
    semana_coseno = math.cos(semana)
    semana_seno = math.sin(semana)
    dia_semana_coseno = math.cos(dia_semana)
    dia_semana_seno = math.sin(dia_semana)

    petroleo_promedio_3meses = float(df_petroleo_promedio_3meses[df_petroleo_promedio_3meses["fecha"] == datetime.strftime(fecha, "%Y-%m-%d")]["petroleo_promedio_3meses"].values[0])
    es_quincena = 1 if (dia == 15 or dia == 30 or dia == 31 ) else 0
    return es_feriado_nacional, es_feriado_local, es_feriado_regional, es_evento, mes_seno, mes_coseno, semana_seno, semana_coseno, dia_semana_seno, dia_semana_coseno, petroleo_promedio_3meses, es_quincena


def obtener_datos_tienda (numero_tienda):
    """Crea diferentes variables con base en una tienda específica.

        Args:
            numero_tienda: int

        Returns:
            ciudad : str
            provincia_estado: str
            tipo_tienda: str
            cluster: int
        """
    ciudad = df_tiendas[df_tiendas["store_nbr"] == numero_tienda]["ciudad"].values[0]
    provincia_estado = df_tiendas[df_tiendas["store_nbr"] == numero_tienda]["provincia_estado"].values[0]
    tipo_tienda = df_tiendas[df_tiendas["store_nbr"] == numero_tienda]["tipo_tienda"].values[0]
    cluster = int(df_tiendas[df_tiendas["store_nbr"] == numero_tienda]["cluster"].values[0])

    return ciudad, provincia_estado, tipo_tienda, cluster


def obtener_datos_articulo (numero_articulo):
    """Crea diferentes variables con base en un número de artículo

        Args:
            numero_articulo: int

        Returns:
            numero_articulo: int
            familia_producto: str
            clase_producto: int
            perecedero: bool
        """
    familia_producto = df_productos[df_productos["numero_articulo"] == numero_articulo]["familia_producto"].values[0]
    clase_producto = int(df_productos[df_productos["numero_articulo"] == numero_articulo]["clase_producto"].values[0])
    perecedero = int(df_productos[df_productos["numero_articulo"] == numero_articulo]["perecedero"].values[0])

    return numero_articulo, familia_producto, clase_producto, perecedero

def mapeo_categoricas(variable, tipo_categoria):
    """mapea una variable con base en los codificadores

        Args:
            variable: str | int
            tipo_categoria: str

        Returns:
            variable mapeada
        """
    if tipo_categoria == "ciudad":
        return dict_ciudad[variable]
    if tipo_categoria == "clase_producto":
        return dict_clase_producto[str(variable)]
    if tipo_categoria == "familia_producto":
        return dict_familia_producto[variable]
    if tipo_categoria == "numero_articulo":
        return dict_numero_articulo[str(variable)]
    if tipo_categoria == "provincia_estado":
        return dict_provincia_estado[variable]
    if tipo_categoria == "tipo_tienda":
        return dict_tipo_tienda[variable]
    return None
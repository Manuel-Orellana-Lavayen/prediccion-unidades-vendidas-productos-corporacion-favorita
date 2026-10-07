# Creado Por Manuel Elias Orellana Lavayen - 2026
"""
    Modelos de Regresión entrenados.

    Este módulo carga los modelos previamente entrenados y almacenados en archivos json y los almacena en una lista
"""

from xgboost import XGBRegressor
from pathlib import Path

# Ruta general
BASE_DIR = Path(__file__).resolve().parent

lista_modelos_cluster = []

for cluster in range(1, 18):
    modelo= XGBRegressor()
    modelo.load_model(fname= f"{BASE_DIR}/cluster_{cluster}.json")

    lista_modelos_cluster.append(modelo)

from pathlib import Path
import urllib.request
import os


BASE_DIR = Path(__file__).resolve().parent
REPO_HF = "https://huggingface.co/Manuel-Orellana-Lavayen/modelos-prediccion-unidades-vendidas-corporacion-favorita/resolve/main"

# RUTAS
MODELOS_CLASIFICACION = {
    BASE_DIR / "Recursos" / "ModelosRegresion" / "cluster_1.json": f"{REPO_HF}/ModelosRegresionCluster/cluster_1.json",
    BASE_DIR / "Recursos" / "ModelosRegresion" / "cluster_2.json": f"{REPO_HF}/ModelosRegresionCluster/cluster_2.json",
    BASE_DIR / "Recursos" / "ModelosRegresion" / "cluster_3.json": f"{REPO_HF}/ModelosRegresionCluster/cluster_3.json",
    BASE_DIR / "Recursos" / "ModelosRegresion" / "cluster_4.json": f"{REPO_HF}/ModelosRegresionCluster/cluster_4.json",
    BASE_DIR / "Recursos" / "ModelosRegresion" / "cluster_5.json": f"{REPO_HF}/ModelosRegresionCluster/cluster_5.json",
    BASE_DIR / "Recursos" / "ModelosRegresion" / "cluster_6.json": f"{REPO_HF}/ModelosRegresionCluster/cluster_6.json",
    BASE_DIR / "Recursos" / "ModelosRegresion" / "cluster_7.json": f"{REPO_HF}/ModelosRegresionCluster/cluster_7.json",
    BASE_DIR / "Recursos" / "ModelosRegresion" / "cluster_8.json": f"{REPO_HF}/ModelosRegresionCluster/cluster_8.json",
    BASE_DIR / "Recursos" / "ModelosRegresion" / "cluster_9.json": f"{REPO_HF}/ModelosRegresionCluster/cluster_9.json",
    BASE_DIR / "Recursos" / "ModelosRegresion" / "cluster_10.json": f"{REPO_HF}/ModelosRegresionCluster/cluster_10.json",
    BASE_DIR / "Recursos" / "ModelosRegresion" / "cluster_11.json": f"{REPO_HF}/ModelosRegresionCluster/cluster_11.json",
    BASE_DIR / "Recursos" / "ModelosRegresion" / "cluster_12.json": f"{REPO_HF}/ModelosRegresionCluster/cluster_12.json",
    BASE_DIR / "Recursos" / "ModelosRegresion" / "cluster_13.json": f"{REPO_HF}/ModelosRegresionCluster/cluster_13.json",
    BASE_DIR / "Recursos" / "ModelosRegresion" / "cluster_14.json": f"{REPO_HF}/ModelosRegresionCluster/cluster_14.json",
    BASE_DIR / "Recursos" / "ModelosRegresion" / "cluster_15.json": f"{REPO_HF}/ModelosRegresionCluster/cluster_15.json",
    BASE_DIR / "Recursos" / "ModelosRegresion" / "cluster_16.json": f"{REPO_HF}/ModelosRegresionCluster/cluster_16.json",
    BASE_DIR / "Recursos" / "ModelosRegresion" / "cluster_17.json": f"{REPO_HF}/ModelosRegresionCluster/cluster_17.json"
}

CODIFICADORES = {
    BASE_DIR / "Recursos" / "PreparacionDatos"/ "CodificadoresCategoricas" / "ciudad.json": f"{REPO_HF}/CodificadoresCategoricas/ciudad.json",
    BASE_DIR / "Recursos" / "PreparacionDatos"/ "CodificadoresCategoricas" / "clase_producto.json": f"{REPO_HF}/CodificadoresCategoricas/clase_producto.json",
    BASE_DIR / "Recursos" / "PreparacionDatos"/ "CodificadoresCategoricas" / "familia_producto.json": f"{REPO_HF}/CodificadoresCategoricas/familia_producto.json",
    BASE_DIR / "Recursos" / "PreparacionDatos"/ "CodificadoresCategoricas" / "numero_articulo.json": f"{REPO_HF}/CodificadoresCategoricas/numero_articulo.json",
    BASE_DIR / "Recursos" / "PreparacionDatos"/ "CodificadoresCategoricas" / "provincia_estado.json": f"{REPO_HF}/CodificadoresCategoricas/provincia_estado.json",
    BASE_DIR / "Recursos" / "PreparacionDatos"/ "CodificadoresCategoricas" / "tipo_tienda.json": f"{REPO_HF}/CodificadoresCategoricas/tipo_tienda.json"
}

BASE_CONOCIMIENTO= {
    BASE_DIR / "Recursos" / "PreparacionDatos" / "BaseConocimiento" / "eventos.csv": f"{REPO_HF}/BaseConocimiento/eventos.csv",
    BASE_DIR / "Recursos" / "PreparacionDatos" / "BaseConocimiento" / "holidays_local.csv": f"{REPO_HF}/BaseConocimiento/holidays_local.csv",
    BASE_DIR / "Recursos" / "PreparacionDatos" / "BaseConocimiento" / "holidays_nacional.csv": f"{REPO_HF}/BaseConocimiento/holidays_nacional.csv",
    BASE_DIR / "Recursos" / "PreparacionDatos" / "BaseConocimiento" / "holidays_regional.csv": f"{REPO_HF}/BaseConocimiento/holidays_regional.csv",
    BASE_DIR / "Recursos" / "PreparacionDatos" / "BaseConocimiento" / "petroleo_promedio_3meses.csv": f"{REPO_HF}/BaseConocimiento/petroleo_promedio_3meses.csv",
    BASE_DIR / "Recursos" / "PreparacionDatos" / "BaseConocimiento" / "productos.csv": f"{REPO_HF}/BaseConocimiento/productos.csv",
    BASE_DIR / "Recursos" / "PreparacionDatos" / "BaseConocimiento" / "testeo_2017.parquet": f"{REPO_HF}/BaseConocimiento/testeo_2017.parquet",
    BASE_DIR / "Recursos" / "PreparacionDatos" / "BaseConocimiento" / "tiendas.csv": f"{REPO_HF}/BaseConocimiento/tiendas.csv"
}


# DESCARGA DE ARCHIVOS JSON, CSV, PARQUET
def descargar_grupo_archivos(grupo_dict: dict):
    for ruta_local, url_descarga in grupo_dict.items():
        # Crea todas las carpetas necesarias para poder guardar el archivo en ruta_local y si ya existen, no las hace
        os.makedirs(os.path.dirname(ruta_local), exist_ok=True)
        if os.path.exists(ruta_local):
            continue

        try:
            urllib.request.urlretrieve(url_descarga,ruta_local)

        except Exception as e:
            raise RuntimeError(
                f"No se pudo descargar el recurso:\n"
                f"Archivo: {ruta_local}\n"
                f"URL: {url_descarga}\n"
                f"Error: {e}"
            ) from e

    return True

# PREPARAR TODO
def preparar_recursos():

    descargar_grupo_archivos(MODELOS_CLASIFICACION)
    descargar_grupo_archivos(CODIFICADORES)
    descargar_grupo_archivos(BASE_CONOCIMIENTO)

    return True

try:
    if not preparar_recursos():
        raise RuntimeError("No se pudieron preparar los recursos.")

except Exception as e:
    print(f"ERROR AL PREPARAR RECURSOS: {e}")
    raise
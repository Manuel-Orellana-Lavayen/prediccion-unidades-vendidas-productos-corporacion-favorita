import pandas as pd
from IPython.display import display
import polars as pl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error, median_absolute_error, root_mean_squared_error,  confusion_matrix, ConfusionMatrixDisplay, classification_report
import xgboost as xgb
import numpy as np
from itertools import product
from polars.lazyframe.engine_config import GPUEngine
config_gpu = GPUEngine(raise_on_fail=True)
pd.options.display.float_format = lambda x: f"{x:,.4f}" if abs(x) < 100 else f"{x:,.0f}"

# Función para metrica NWEMSLE

def nwrmsle(y_true, y_pred, perecedero):
    """
    Carcula la metrica NWEMSLE

    Args:
        y_true: Array de valores reales
        y_pred: Array de valores predichos
        perecedero: Array de boleanos sobre si el producto es o no es perecedero

    Returns:
        metrica NWEMSLE
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    perecedero = np.asarray(perecedero)

    # Se reemplazan llos valores negativos por 0
    y_pred = np.maximum(y_pred, 0)

    pesos = np.where(perecedero == 1, 1.25, 1.0)

    error_log = (np.log1p(y_pred) - np.log1p(y_true)) ** 2

    return np.sqrt(np.sum(pesos * error_log) / np.sum(pesos))

# Función Para Entrenar diferentes modelos basados en regresión según una categoria y ver sus metricas

def modelos_regresion(lista_clusters : list, df : pl.lazyframe, periodos_entrenamiento: list, periodo_testeto: list,pesos_perecedero: bool = False):
    """
    Entrena diferentes modelos de regresión usando XGBRegressor de XGBOOST con una respectiva configuración de sus hiperparámetros

    Args:
        lista_clusters: Lista de clusters a entrenar
        df : LazyFrame de polars
        periodos_entrenamiento: Lista de años para entrenar modelo
        periodo_testeto: lista de años para predecir modelo
        pesos_perecedero: Booleano si se quiere aplicar en el entrenamiento un peso de penalización a los productos perecederos, frente a los no perecederos

    Returns:
        Diccionario con nombre del cluster y modelo respectivo a ese cluster
    """

    diccionario_modelos = {}
    for cluster in lista_clusters:
        print(f"------------------------cluster {cluster}------------------------")

        X_entrenamiento_cuda = df.filter(pl.col("año").is_in(periodos_entrenamiento)).filter(pl.col("cluster") == cluster).drop(["cluster", "año", "unidades_vendidas"]).collect(engine=config_gpu).to_arrow()
        y_entrenamiento_cuda = df.select(["año", "unidades_vendidas", "cluster", "perecedero"]).filter(pl.col("año").is_in(periodos_entrenamiento)).filter(pl.col("cluster") == cluster).drop(["año", "cluster"]).collect(engine=config_gpu).to_arrow()

        X_testeo_no_cuda = df.filter(pl.col("año").is_in(periodo_testeto)).filter(pl.col("cluster") == cluster).drop(["cluster", "año", "unidades_vendidas"]).collect(engine="streaming")
        y_testeo_no_cuda = df.select(["año", "unidades_vendidas", "cluster", "perecedero"]).filter(pl.col("año").is_in(periodo_testeto)).filter(pl.col("cluster") == cluster).drop(["año", "cluster"]).collect(engine="streaming")

        pesos = np.where(y_entrenamiento_cuda["perecedero"].to_numpy() == 1,1.25,1.0)
        modelo = xgb.XGBRegressor(
            enable_categorical=True,
            tree_method="hist",
            device="cuda",
            learning_rate = 0.1,
            max_depth = 8,
            tweedie_variance_power = 1.5,
            max_bin=256,
            seed=42,
            objective="reg:tweedie",
            max_delta_step=5,
            gamma=3,
            min_child_weight= 6000,
            reg_lambda=70,
            feature_types=["c", "c", "c", "c", "c", "c", "c", "q", "q", "q", "q", "q", "q", "q", "q", "q", "q", "q","q", "q", "q", "q", "q"])

        if pesos_perecedero == True:
            modelo.fit(X= X_entrenamiento_cuda, y= y_entrenamiento_cuda["unidades_vendidas"], sample_weight=pesos)
        else:
            modelo.fit(X=X_entrenamiento_cuda, y=y_entrenamiento_cuda["unidades_vendidas"])

        modelo.get_booster().set_param({"device": "cpu"}) # Obligamos a el modelo a usar CPU para las predicciones (Evitamos posible saturación de memoria vram)

        predicciones = modelo.predict(X= X_testeo_no_cuda.to_pandas())

        error_absoluto_medio            = mean_absolute_error(y_pred=predicciones, y_true=y_testeo_no_cuda["unidades_vendidas"])
        raiz_error_cuadratico_medio     = root_mean_squared_error(y_pred=predicciones, y_true=y_testeo_no_cuda["unidades_vendidas"])
        r2                              = r2_score(y_pred=predicciones, y_true=y_testeo_no_cuda["unidades_vendidas"])
        error_maximo                    = max_error(y_pred=predicciones, y_true=y_testeo_no_cuda["unidades_vendidas"])
        mediana_error_absoluto          = median_absolute_error(y_pred=predicciones, y_true=y_testeo_no_cuda["unidades_vendidas"])
        nwrmsle_metrica                 = nwrmsle(y_true=y_testeo_no_cuda["unidades_vendidas"].to_numpy(),y_pred=predicciones,perecedero=y_testeo_no_cuda["perecedero"].to_numpy())

        print("Descripción de datos")
        display(y_testeo_no_cuda.to_series().describe())

        print("Métricas")
        diccionario_metricas = {
            "Predicción Maxima"    : max(predicciones),
            "Predicción Minima"    : min(predicciones),
            "Ventas Reales"        : y_testeo_no_cuda["unidades_vendidas"].sum(),
            "Ventas Predichas"     : predicciones.sum(),
            "Mean Absolute Error"  : error_absoluto_medio,
            "Median Absolute Error": mediana_error_absoluto,
            "RMSE"                 : raiz_error_cuadratico_medio,
            "Max Error"            : error_maximo,
            "R2"                   : r2,
            "NWRMSLE"              : nwrmsle_metrica
        }
        display(pd.DataFrame(data= diccionario_metricas, index= ["Resultados"]).T)

        print("Rendimiento por bloques de unidades vendidas")
        bloques = [
            ("0", 0, 0),
            ("1-5", 1, 5),
            ("6-20", 6, 20),
            ("21-50", 21, 50),
            ("51-100", 51, 100),
            ("101-500", 101, 500),
            (">500", 501, np.inf)
        ]
        y_real = y_testeo_no_cuda["unidades_vendidas"].to_pandas()
        y_real_perecedero = y_testeo_no_cuda["perecedero"].to_pandas()

        resultados_bloques = []
        for nombre_bloque, minimo, maximo in bloques:
            mascara     = ((y_real >= minimo) &(y_real <= maximo))
            y_bloque    = y_real[mascara]
            perecedero_bloque = y_real_perecedero[mascara]
            pred_bloque = predicciones[mascara]

            if len(y_bloque) == 0:
                continue

            resultados_bloques.append({
                "bloque"          : nombre_bloque,
                "cantidad"        : len(y_bloque),
                "porcentaje"      : len(y_bloque) / len(y_real) * 100,
                "MAE"             : mean_absolute_error(y_bloque, pred_bloque),
                "RMSE"            : np.sqrt(mean_squared_error(y_bloque, pred_bloque)),
                "Ventas Reales"   : y_bloque.sum(),
                "Ventas Predichas": pred_bloque.sum(),
                "NWRMSLE"         : nwrmsle(y_true=y_bloque, y_pred=pred_bloque, perecedero=perecedero_bloque.to_numpy())
            })
        display(pd.DataFrame(data = resultados_bloques).set_index("bloque").T)

        diccionario_modelos[cluster] = modelo

    return diccionario_modelos

# Función Para Entrenar diferentes modelos basados en clasificación segun una categoria y ver sus metricas

def modelos_experimentales_clasificacion(lista_clusters : list, df : pl.lazyframe, periodos_entrenamiento: list, periodo_testeto: list):
    """
    Entrena diferentes modelos de clasificación usando XGBClassifier de XGBOOST con una respectiva configuración de sus hiperparámetros,
    el modelo de clasificación busca saber si van a existir una venta grande o no. Tiene como propósito mejorar el modelo en sus predicciones a valores altos como modelo de dos etapas

    Args:
        lista_clusters: Lista de clusters a entrenar
        df : LazyFrame de polars
        periodos_entrenamiento: Lista de años para entrenar modelo
        periodo_testeto: lista de años para predecir modelo

    Returns:
        Diccionario con nombre del cluster y modelo respectivo a ese cluster
    """
    diccionario_modelos = {}
    for cluster in lista_clusters:
        print(f"------------------------cluster {cluster}------------------------")
        X_entrenamiento_cuda = df.filter(pl.col("año").is_in(periodos_entrenamiento)).filter(pl.col("cluster") == cluster).drop(["cluster", "año", "venta_grande"]).collect( engine=config_gpu).to_arrow()
        y_entrenamiento_cuda = df.select(["año", "venta_grande", "cluster"]).filter(pl.col("año").is_in(periodos_entrenamiento)).filter(pl.col("cluster") == cluster).drop(["año", "cluster"]).collect(engine=config_gpu).to_arrow()

        X_testeo_no_cuda = df.filter(pl.col("año").is_in(periodo_testeto)).filter(pl.col("cluster") == cluster).drop(["cluster", "año", "venta_grande"]).collect(engine="streaming")
        y_testeo_no_cuda = df.select(["año", "venta_grande", "cluster"]).filter(pl.col("año").is_in(periodo_testeto)).filter(pl.col("cluster") == cluster).drop(["año", "cluster"]).collect(engine="streaming")

        modelo = xgb.XGBClassifier(
            enable_categorical=True,
            tree_method="hist",
            device="cuda",
            eval_metric="aucpr",
            learning_rate=0.05,
            max_depth=6,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            max_delta_step=2,
            feature_types=["c", "c", "c", "c", "c", "c", "c", "q", "q", "q", "q", "q", "q", "q", "q", "q", "q", "q","q", "q", "q", "q", "q"]
        )

        modelo.fit(X=X_entrenamiento_cuda, y=y_entrenamiento_cuda)

        modelo.get_booster().set_param({"device": "cpu"})

        # Modificando predicciones, donde a la más minima duda de la clase 1, lo convierta en clase 0
        predicciones_clase1 = modelo.predict_proba(X=X_testeo_no_cuda.to_pandas())[:, 1]
        predicciones = np.where(predicciones_clase1 <= 0.001, 0, 1)

        matriz_confusion = confusion_matrix(y_true=y_testeo_no_cuda, y_pred=predicciones)
        matriz_confusion = ConfusionMatrixDisplay(matriz_confusion)
        reporte_clasificacion = classification_report(y_true=y_testeo_no_cuda, y_pred=predicciones, output_dict=True)
        reporte_clasificacion = pd.DataFrame(reporte_clasificacion)

        fig, axs = plt.subplots(ncols=2, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 1.5]})

        # Matriz
        matriz_confusion.plot(ax=axs[0], cmap="Blues", values_format="d", colorbar = False)
        axs[0].set_title(f"Matriz de Confusión", fontsize=12, pad=10)

        # Reporte
        axs[1].axis("off")
        tabla1 = axs[1].table(
            cellText=reporte_clasificacion.values.round(3),  # Añadido .round(3) para consistencia
            colLabels=reporte_clasificacion.columns,
            rowLabels=reporte_clasificacion.index,
            cellLoc="center",
            loc="center",
        )

        tabla1.scale(1, 1.4)
        axs[1].set_title(f"Reporte de Clasificación", fontsize=12, pad=10)

        plt.tight_layout()
        plt.show()
        # Guardando modelo en el diccionario
        diccionario_modelos[cluster] = modelo

    return diccionario_modelos

def modelos_experimentales_clasificacion_2(lista_clusters : list, df : pl.lazyframe, periodos_entrenamiento: list, periodo_testeto: list):
    """
        Entrena diferentes modelos de clasificación usando XGBClassifier de XGBOOST con una respectiva configuración de sus hiperparámetros,
        el modelo de clasificación busca saber si va a existir una venta o no. Tiene como propósito mejorar el modelo en sus predicciones a valores bajos.

        Args:
            lista_clusters: Lista de clusters a entrenar
            df : LazyFrame de polars
            periodos_entrenamiento: Lista de años para entrenar modelo
            periodo_testeto: lista de años para predecir modelo

        Returns:
            Diccionario con nombre del cluster y modelo respectivo a ese cluster
        """
    diccionario_modelos = {}
    for cluster in lista_clusters:
        print(f"------------------------cluster {cluster}------------------------")
        X_entrenamiento_cuda = df.filter(pl.col("año").is_in(periodos_entrenamiento)).filter(pl.col("cluster") == cluster).drop(["cluster", "año", "Se vende"]).collect( engine=config_gpu).to_arrow()
        y_entrenamiento_cuda = df.select(["año", "Se vende", "cluster"]).filter(pl.col("año").is_in(periodos_entrenamiento)).filter(pl.col("cluster") == cluster).drop(["año", "cluster"]).collect(engine=config_gpu).to_arrow()

        X_testeo_no_cuda = df.filter(pl.col("año").is_in(periodo_testeto)).filter(pl.col("cluster") == cluster).drop(["cluster", "año", "Se vende"]).collect(engine="streaming")
        y_testeo_no_cuda = df.select(["año", "Se vende", "cluster"]).filter(pl.col("año").is_in(periodo_testeto)).filter(pl.col("cluster") == cluster).drop(["año", "cluster"]).collect(engine="streaming")

        modelo = xgb.XGBClassifier(
            enable_categorical=True,
            tree_method="hist",
            device="cuda",
            eval_metric="aucpr",
            learning_rate=0.05,
            max_depth=6,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            max_delta_step=2,
            feature_types=["c", "c", "c", "c", "c", "c", "c", "q", "q", "q", "q", "q", "q", "q", "q", "q", "q", "q","q", "q", "q", "q", "q"]
        )

        modelo.fit(X=X_entrenamiento_cuda, y=y_entrenamiento_cuda)

        modelo.get_booster().set_param({"device": "cpu"})

        # Modificando predicciones, donde a la más minima duda de la clase 1, lo convierta en clase 0
        predicciones_clase1 = modelo.predict_proba(X=X_testeo_no_cuda.to_pandas())[:, 1]
        predicciones = np.where(predicciones_clase1 <= 0.20, 0, 1)

        matriz_confusion = confusion_matrix(y_true=y_testeo_no_cuda, y_pred=predicciones)
        matriz_confusion = ConfusionMatrixDisplay(matriz_confusion)
        reporte_clasificacion = classification_report(y_true=y_testeo_no_cuda, y_pred=predicciones, output_dict=True)
        reporte_clasificacion = pd.DataFrame(reporte_clasificacion)

        fig, axs = plt.subplots(ncols=2, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 1.5]})

        # Matriz
        matriz_confusion.plot(ax=axs[0], cmap="Blues", values_format="d", colorbar = False)
        axs[0].set_title(f"Matriz de Confusión", fontsize=12, pad=10)

        # Reporte
        axs[1].axis("off")
        tabla1 = axs[1].table(
            cellText=reporte_clasificacion.values.round(3),  # Añadido .round(3) para consistencia
            colLabels=reporte_clasificacion.columns,
            rowLabels=reporte_clasificacion.index,
            cellLoc="center",
            loc="center",
        )

        tabla1.scale(1, 1.4)
        axs[1].set_title(f"Reporte de Clasificación", fontsize=12, pad=10)

        plt.tight_layout()
        plt.show()
        # Guardando modelo en el diccionario
        diccionario_modelos[cluster] = modelo

    return diccionario_modelos


def modelos_hiperparametros_regresion (lista_clusters : list, df : pl.lazyframe, periodos_entrenamiento: list, periodo_testeto: list, param_grid:dict):
    """
        Entrena diferentes modelos de regresión usando XGBRegressor de XGBOOST con un grid de hiperparámetros que permite
        realizar multiples configuraciones diferentes para tratar de encontrar la mejor combinación para respectivas métricas
        y de esa manera escoger los mejores para el entrenamiento final de los modelos.

        Args:
            lista_clusters: Lista de clusters a entrenar
            df : LazyFrame de polars
            periodos_entrenamiento: Lista de años para entrenar modelo
            periodo_testeto: lista de años para predecir modelo
            param_grid: Diccionario de diferentes configuraciones para hiperparametros del modelo

        Returns:
            No devuelve ningún valor, solo imprime las métricas y configuraciones para cada cluster
        """
    for cluster in lista_clusters:

        print(f"------------------------cluster {cluster}------------------------")

        X_entrenamiento_cuda = df.filter(pl.col("año").is_in(periodos_entrenamiento)).filter(pl.col("cluster") == cluster).drop(["cluster", "año", "unidades_vendidas"]).collect(engine=config_gpu).to_arrow()
        y_entrenamiento_cuda = df.select(["año", "unidades_vendidas", "cluster"]).filter(pl.col("año").is_in(periodos_entrenamiento)).filter(pl.col("cluster") == cluster).drop(["año", "cluster"]).collect(engine=config_gpu).to_arrow()

        X_testeo_no_cuda = df.filter(pl.col("año").is_in(periodo_testeto)).filter(pl.col("cluster") == cluster).drop(["cluster", "año", "unidades_vendidas"]).collect(engine="streaming")
        y_testeo_no_cuda = df.select(["año", "unidades_vendidas", "cluster", "perecedero"]).filter(pl.col("año").is_in(periodo_testeto)).filter(pl.col("cluster") == cluster).drop(["año", "cluster"]).collect(engine="streaming")

        display(y_testeo_no_cuda.to_series().describe())
        lista_resultados = []

        for max_delta_step, learning_rate, gamma, max_depth, min_child_weight, lambda_param, tweedie_variance_power in product(
                                                            param_grid["max_delta_step"],
                                                            param_grid["learning_rate"],
                                                            param_grid["gamma"],
                                                            param_grid["max_depth"],
                                                            param_grid["min_child_weight"],
                                                            param_grid["lambda_param"],
                                                            param_grid["tweedie_variance_power"]

        ):

            modelo = xgb.XGBRegressor(
                enable_categorical=True,
                tree_method="hist",
                device="cuda",
                seed = 42,
                objective = "reg:tweedie",
                max_delta_step= max_delta_step,
                learning_rate = learning_rate,
                gamma = gamma,
                max_depth = max_depth,
                min_child_weight = min_child_weight,
                reg_lambda = lambda_param,
                max_bin = 256,
                tweedie_variance_power = tweedie_variance_power,
                feature_types = ["c", "c", "c", "c", "c", "c", "c", "q", "q", "q", "q", "q", "q", "q", "q", "q", "q", "q", "q", "q", "q", "q", "q"]
                )

            modelo.fit(X= X_entrenamiento_cuda, y= y_entrenamiento_cuda)

            modelo.get_booster().set_param({"device": "cpu"})

            predicciones = modelo.predict(X= X_testeo_no_cuda.to_pandas())

            error_absoluto_medio = mean_absolute_error(y_pred= predicciones, y_true= y_testeo_no_cuda["unidades_vendidas"].to_numpy())
            raiz_error_cuadratico_medio = root_mean_squared_error(y_pred= predicciones, y_true= y_testeo_no_cuda["unidades_vendidas"].to_numpy())
            r2 = r2_score(y_pred= predicciones, y_true= y_testeo_no_cuda["unidades_vendidas"].to_numpy())
            nwrmsle_metrica = nwrmsle(y_true=y_testeo_no_cuda["unidades_vendidas"].to_numpy(), y_pred=predicciones, perecedero=y_testeo_no_cuda["perecedero"].to_numpy())

            lista_resultados.append({"max_delta_step": max_delta_step,
                                            "learning_rate": learning_rate,
                                            "gamma": gamma,
                                            "max_depth": max_depth,
                                            "min_child_weight": min_child_weight,
                                            "reg_lambda": lambda_param,
                                            "tweedie_variance_power": tweedie_variance_power,
                                            "Mean Absolute Error": error_absoluto_medio,
                                            "RMSE": raiz_error_cuadratico_medio,
                                            "R2": r2,
                                            "NWRMSLE": nwrmsle_metrica})

        df_lista_resultados = pd.DataFrame(lista_resultados)
        print()
        print("Mejor Combinación --- Mean Absolute Error")
        print()
        print(df_lista_resultados.sort_values("Mean Absolute Error").iloc[0])
        print()
        print("Mejor Combinación --- RMSE")
        print()
        print(df_lista_resultados.sort_values("RMSE").iloc[0])
        print()
        print("Mejor Combinación --- R2")
        print()
        print(df_lista_resultados.sort_values("R2", ascending=False).iloc[0])
        print()
        print("Mejor Combinación --- NWRMSLE")
        print()
        print(df_lista_resultados.sort_values("NWRMSLE").iloc[0])

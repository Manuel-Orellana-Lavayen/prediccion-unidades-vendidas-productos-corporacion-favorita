import pandas as pd

def resumen_dataset (dataframe: pd.DataFrame, columnas_excluir: list):
    """
    Resume un dataframe excluyendo columnas específicas, devuelve informacion por columna como:
    * Cantidad de filas
    * Tipo de dato
    * Cantidad de datos unicos
    * Los datos unicos
    * Distribución de los datos

    Args:
        dataframe: DataFrame de  pandas,
        columnas_excluir: lista de nombres de columnas a exluir del analisis

    Returns:
        No retorna nigun valor. Solo imprime
    """
    columnas = dataframe.columns
    columnas = columnas.drop(columnas_excluir)

    diccionario_elementos_columna = {}

    for columna in columnas:
        cantidad_filas = len(dataframe[columna])
        tipo_dato = dataframe[columna].dtypes
        cantidad_datos_unicos = dataframe[columna].nunique()
        datos_unicos = dataframe[columna].unique()
        distribucion_datos = dataframe[columna].value_counts(normalize=True) * 100

        diccionario_elementos_columna[columna] = {
            "Número de filas": cantidad_filas,
            "Tipo de dato": tipo_dato,
            "Cantidad de datos únicos": cantidad_datos_unicos,
            "Datos únicos": list(datos_unicos),
            "Distribución de datos (%)": distribucion_datos,
        }

    print("\n" + "═" * 60)
    print(" RESUMEN DEL DATASET ".center(60, "═"))
    print("═" * 60)

    for columna, info in diccionario_elementos_columna.items():
        print(f" COLUMNA: {columna.upper()}")
        print("─" * 60)
        print(f"  • Número de filas:        {info['Número de filas']}")
        print(f"  • Tipo de dato:           {info['Tipo de dato']}")
        print(f"  • Cantidad datos únicos:  {info['Cantidad de datos únicos']}")

        unicos = info["Datos únicos"]
        if len(unicos) > 5:
            unicos_str = f"{unicos[:5]} ... (total {len(unicos)})"
        else:
            unicos_str = str(unicos)
        print(f"  • Valores únicos:         {unicos_str}")
        print("  • Distribución de datos:")
        distribucion_fmt = info["Distribución de datos (%)"].round(2)
        for valor, pct in distribucion_fmt.items():
            print(f"      - {str(valor):<25} : {pct:>6.2f}%")

        print("─" * 60)
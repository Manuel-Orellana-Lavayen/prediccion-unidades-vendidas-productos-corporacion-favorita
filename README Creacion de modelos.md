# clasificacion-multicategoria-resenas-espanol

## Desarrollo de modelos

### librerías
Polars	
Pandas	
NumPy	
PyArrow	
scikit-learn	
SciPy	
statsmodels	
Pingouin	
Matplotlib	
Seaborn	
xgboost-gpu 
cudf-polars-cu12 
watermark 

### Descripción
Modelo de Machine Learning para predicción a corto plazo (dia siguiente) de unidades vendidas por producto para todos los establecimientos comerciales de Corporación Favorita. Modelo entrenado con datos historicos de 2013 a 2017.

### Objetivo
Crear una aplicación pueda predecir a corto plazo las unidades vendidas por producto para todos los establecimientos comerciales de Corporación Favorita.

```
└── Creacion de Modelos/
    │
    ├── Analisis y entrenamiento de modelos/
    │   │
    │   ├── 1-Concatenando Datasets.ipynb
    │   │     └── Archivos ipynb que concatena los datasets uno por uno para tener control de la memoria RAM usada.
    │   ├── 2-Imputacion Nulos Datasets Concatenados.ipynb
    │   │     └── Se hace la imputación de los valores nulos del dataset concatenado resultante
    │   ├── 3-Ingenieria Caracteristicas.ipynb
    │   │     └── Se eliminan valores negativos de la columna de unidades vendidas para convertirlos en 0. Además se crean columnas útiles para la fase del entrenamiento.
    │   ├── 4-Analisis Exploratorio Dataset Concatenado.ipynb
    │   │     └── Se analizan algunas columnas del dataset concatenado y limpio. Además se busca visualizar la correlación de variables (No proporcionan mucha información)
    │   ├── 5-Modelos-Experimentales.ipynb
    │   │     └── Se entrenan distintos modelos con diferentes hiperparametros. En base a sus métricas se va eligiendo los mejores parámetros para el entrenamiento final
    │   ├── 6-Modelos Finales.ipynb
    │   │     └── Se entrenan los modelos fínales para hacer el análisis sus métricas de manera general y por bloques de unidades vendidas.
    │   ├── FuncionesModelos.py
    │   │     └── Archivo .py donde se guardan las funciones para entrenamiento de modelos.
    │   │
    │   ├── Datasets/
    │   │   ├── Concatenados/
    │   │   │   ├── Limpios/
    │   │   │   │   └── Carpeta donde se guardan los datasets limpios 
    │   │   │   └── Sucios/
    │   │   │       └── Carpeta donde se guardan los datasets que aún le falta limpieza
    │   │   │
    │   │   ├── Modificados/
    │   │   │   ├── dataset holidays_events.ipynb
    │   │   │   │   └── Archivos que buscan crear datasets modificados en esta misma carpeta.
    │   │   │   ├── dataset oil.ipynb
    │   │   │   └── dataset train.ipynb
    │   │   │
    │   │   └── Originales/
    │   │       ├── Exploración Individual Datasets/
    │   │       │   ├── FuncionResumenDataset.py
    │   │       │   ├── holidays_events.ipynb
    │   │       │   ├── items.ipynb
    │   │       │   ├── oil.ipynb
    │   │       │   ├── stores.ipynb
    │   │       │   ├── test.ipynb
    │   │       │   ├── train.ipynb
    │   │       │   └── transactions.ipynb
    │   │       └── Datasets originales ...
    │   │
    │   ├── Modelos Finales/
    │   │   └── Carpeta donde se guardan los 17 modelos finales (1 por cada cluster)
    │   │
    │   └── Codificadores Categoricas/
    │       └── Carpeta donde se guardan los codificadores categoricos para preparación de datos.
    │     
    ├── CreacionBaseConocimiento/
    │   ├── BaseConocimiento/
    │   │   └── Lugar donde se guardan las bases de conocimiento creadas con los archivos de la carpeta "Creadores de base de conocimiento"
    │   │
    │   └── Creadores de base de conocimiento/
    │       ├── holidays_events.ipynb
    │       ├── items.ipynb
    │       ├── oil.ipynb
    │       ├── stores.ipynb
    │       └── testeo_2017.ipynb
    │   
    ├── Dockerfile
    │   └── Dockfile basado en imagen de ubuntu
    │
    ├── pixi.lock
    │
    ├── pixi.toml
    │   └── Librerias necesarias (Para crear un entorno .pixi)
    │
    ├── .devcontainer/
    │   └── devcontainer.json
    │       └── Archivo devcontainer.json para crear contenedor en base a la imagen creada con el dockfile y hacer desarrollo en contenedores con sincronización local
    │
    └── requirements.txt
        └── Librerias necesarias (Para crear un entorno .venv)
```

## Instalación y ejecución de Creación de Modelos

### Fase 1

### Fase 2

1. Crear una imagen del contenedor usando Dockerfile
2. Abrir la carpeta donde estén todos los archivos de Creacion de Modelos
3. En la terminal escribir el siguiente comando para crear un contenedor conectado a la carpeta local en base a la imagen ya creada:
docker run -it --gpus all --cpus="10" --memory="14.5g" --name nombre_contenedor -v "$(pwd):/espacio_trabajo" nombre_imagen
4. Dentro del contenedor copiar los archivos pixi.toml y pixi.lock a la carpeta llamada "entorno_virtual"
5. Abrir una terminal para esa carpeta y ejecutar el siguiente comando para instalar en entorno virtual .pixi  -> "pixi install"
6. Configurar el interprete dentro de nuestro contenedor

## Autor
Manuel Elias Orellana Lavayen 

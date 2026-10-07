# clasificacion-multicategoria-resenas-espanol

## Desarrollo de modelos

### librerías
- python
- cuml
- cupy
- torch
- sentence-transformers
- symspellpy
- scikit-learn
- pandas
- spacy
- datasets
- scipy
- matplotlib
- es-core-news-md
- joblib 
- watermark 

### Descripción
Modelos de Machine Learning para clasificar reseñas en español en tres categorías de sentimiento: positiva, neutral y negativa.

### Objetivo
Crear una aplicación que clasifique las reseñas de un documento CSV en 3 clases y muestre las distribución de cada una de sus clases en un gráfico de pastel

### Dataset
amazon_reviews_multi_es -> Dataset de 200.000 reseñas en español de productos, las reseñas están clasificadas en 5 clases. Además de 5.000 Reseñas de testeo y 5.000 Reseñas de validación

```
├── Creacion de Modelos/
│   │
│   ├── API/
│   │   │
│   │   ├── Analisis de datos y entrenamiento de modelos/
│   │   │   ├── Datasets
│   │   │   │   └── Carpeta donde se guardan los datasets descargados 
│   │   │   └── Datasets Limpios y corregidos
│   │   │   │   └── Carpeta donde se guardan los datasets limpios
│   │   │   ├── funciones_limpieza_ensemble
│   │   │   │   └── funciones_limpieza.py
│   │   │   │       └── Modulo con funciones de limpieza, útil al momento de crear el modelo Ensamble
│   │   │   └── matrices y embeddings
│   │   │   │    └── No Ortografia
│   │   │   │    │   ├── Embeddings
│   │   │   │    │   │   └── Carpeta donde se guarda las vectorizaciones de los datasets (Embeddings)
│   │   │   │    │   └── Matrices TF-IDF
│   │   │   │    │      ├── sublinear_false
│   │   │   │    │      │   ├── ngram_2
│   │   │   │    │      │   │   ├── 10000
│   │   │   │    │      │   │   │   └── Carpetas donde se guarda las vectorizaciones de los datasets y su respectivo vectorizador (TF-IDF)
│   │   │   │    │      │   │   ├── 20000
│   │   │   │    │      │   │   ├── 30000
│   │   │   │    │      │   │   ├── 40000
│   │   │   │    │      │   │   └── 50000
│   │   │   │    │      │   ├── ...
│   │   │   │    │      └── sublinear_true
│   │   │   │    │          └── ...
│   │   │   │    └── Ortografia
│   │   │   │    │   └──...
│   │   │   ├── Modelos Finales
│   │   │   │   └── Carpeta donde se guardan los modelos resultantes del notebook "4_modelos_finales.ipynb"
│   │   │   └── Recursos de procesamiento de texto
│   │   │   │   └── Carpeta donde se colocan los recursos de preprocesamiento de texto que se obtienen de "Descarga de modelos recursos.ipynb"
│   │   │   ├── Resultados Modelos Experimentos
│   │   │   │   └── Carpeta donde se guarda las tablas de los modelos experimentales que se entrenan en el notebook "3_modelos_experimentales.ipynb"
│   │   │   └── Y
│   │   │       └── Carpeta donde se guardan las etiquetas de los datasets
│   │   │   └── 1_limpieza.ipynb
│   │   │   │   └── Limpieza de los datasets
│   │   │   ├── 2_vectorizacion.ipynb
│   │   │   │   └── Vectorización de los datasets limpios
│   │   │   └── 3_modelos_experimentales.ipynb
│   │   │       └── Entrenamiento de 244 modelos experimentales para obtener la mejor configuración 
│   │   │   └── 4_modelos_finales.ipynb
│   │   │   │   └── Entrenamiento de los modelos finales basándonos en la mejor configuración obtenida en el notebook "3_modelos_experimentales.ipynb"
│   │   │   ├── 5_modelo_ensemble.ipynb
│   │   │   │   └── Creación de modelo Ensamble
│   │   │   └── Instrucciones.txt
│   │   │       └── Instrucciones de los pasos a realizar para utilizar los modelos resultantes en la aplicación
│   │   ├── Descargando Dataset/
│   │   │   ├── Descargando Dataset.ipynb
│   │   │   │   └── Descarga de datasets
│   │   │   └── Instrucciones.txt
│   │   │       └── Instrucciones de donde colocar los datasets
│   │   ├── Descargando Recursos/
│   │   │   ├── Descarga de modelos recursos.ipynb
│   │   │   │   └── Descarga de recursos (vectorizador Embeddings y limpiador de palabras)
│   │   │   ├── Recursos de procesamiento de texto
│   │   │   │   └── Carpeta donde se guardan los recursos descargados
│   │   │   └── Instrucciones.txt
│   │   │       └── Instrucciones de donde colocar los recursos 
│   │   ├── Dockerfile
│   │   │   └── Dockfile basado en imagen de ubuntu
│   │   ├── pixi.lock
└── └── └── pixi.toml
            └── Librerias necesarias (Para crear un entorno .pixi)
```

## Instalación y ejecución de Creación de Modelos

1. Crear una imagen del contenedor usando Dockerfile
2. Abrir la carpeta donde estén todos los archivos de Creacion de Modelos
3. En la terminal escribir el siguiente comando para crear un contenedor conectado a la carpeta local en base a la imagen ya creada:
docker run -it --gpus all --cpus="10" --memory="14.5g" --name nombre_contenedor -v "$(pwd):/espacio_trabajo" nombre_imagen
4. Dentro del contenedor copiar los archivos pixi.toml y pixi.lock a la carpeta llamada "entorno_virtual"
5. Abrir una terminal para esa carpeta y ejecutar el siguiente comando para instalar en entorno virtual .pixi  -> "pixi install"
6. Configurar el interprete dentro de nuestro contenedor

## Autor
Manuel Elias Orellana Lavayen 

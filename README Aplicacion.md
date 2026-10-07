# clasificacion-multicategoria-resenas-espanol

## Aplicacion

### librerías
- scikit-learn
- pandas
- numpy
- spacy
- sentence-transformers
- fastapi
- uvicorn
- pydantic
- cloudpathlib
- joblib
- plotly
- requests
- streamlit

```
├── Aplicación/
│   │
│   ├── API/
│   │   │
│   │   ├── CreacionPipelines/
│   │   │   │
│   │   │   ├── Pipelines/
│   │   │   │   └── usar_pipelines.py
│   │   │   │       └─ Funciones para utilizar los pipelines de predicción.
│   │   │   │
│   │   │   ├── Recursos/
│   │   │   │   │
│   │   │   │   ├── Clases/
│   │   │   │   │   ├── ClasesAdaptadorasEnsamble.py
│   │   │   │   │   ├── ClasesEnvoltorio.py
│   │   │   │   │   └── ClasesPredictoras.py
│   │   │   │   │       └─ Clases auxiliares utilizadas por los modelos.
│   │   │   │   │
│   │   │   │   ├── EstilosGraficosPlotly/
│   │   │   │   │   └── EstilosPlotly.py
│   │   │   │   │       └─ Define estilos reutilizables para gráficos Plotly.
│   │   │   │   │
│   │   │   │   ├── Limpieza/
│   │   │   │   │   └── FuncionesLimpieza.py
│   │   │   │   │       └─ Funciones utilizadas para el preprocesamiento del texto.
│   │   │   │   │
│   │   │   │   ├── ModelosClasificacion/
│   │   │   │   │   └── Modelos.py
│   │   │   │   │       └─ Carga los modelos de clasificación entrenados.
│   │   │   │   │
│   │   │   │   └── Vectorizacion/
│   │   │   │       └── Vectorizadores.py
│   │   │   │           └─ Carga los recursos necesarios para vectorizar textos.
│   │   │   │
│   │   │   ├── Crear Pipelines.py
│   │   │   │   └─ Construye los pipelines utilizando modelos, vectorizadores y funciones de procesamiento.
│   │   │   └── Instalando Recursos.py
│   │   │       └─ Instala los recursos desde un repositorio de huggingface en caso de que falten.
│   │   │
│   │   ├── api_aplicacion.py
│   │   │   └─ Define la API y sus endpoints mediante FastAPI.
│   │   │
│   │   └── FuncionesApi.py
│   │       └─ Contiene las funciones finales utilizadas por la API.
│   │
│   └── Interfaz Gráfica/
│       │
│       ├── .streamlit/
│       │   └─ Configuración de la interfaz de Streamlit.
│       │
│       ├── static/
│       │   └─ Fuentes utilizadas por la interfaz.
│       │
│       ├── Interfaz_aplicacion.py
│       │   └─ Código principal de la interfaz gráfica.
│       │
│       └── ArchivoEjemplo.csv
│           └─ Archivo de ejemplo para probar la aplicación.
│
├── requirements_aplicación.txt
│    └─ Dependencias necesarias para ejecutar la aplicación.
├── Dockerfile_aplicacion
│    └─ Dockfile para ejecutar la aplicación.
├── Dockerfile_modificacion
│    └─ Dockfile para modificar algunos aspectos de la aplicación (Especialmente para crear pipelines).
└── start.sh
     └─ Comandos que ejecutan diferentes archivos del proyecto para encender la aplicación cuando se usa el Dockerfile_aplicacion
```


### Instalación y ejecución de Aplicación
1. Crear una imagen del contenedor usando Dockerfile_aplicacion
2. Crear un contenedor de la imagen dejando expuesto el puerto indicado en el Dockerfile_aplicacion
3. Abrir la interfaz de streamlit con el localhost en cualquier buscador

## Autor
Manuel Elias Orellana Lavayen 

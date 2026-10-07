# prediccion-unidades-vendidas-productos-corporacion-favorita

## Aplicacion

### librerías
- Streamlit
- FastAPI
- Uvicorn
- XGBoost
- SHAP	
- Polars	
- Pandas	
- SciPy	
- Matplotlib	
- Plotly	
- Joblib	
- Requests	
- Pydantic	

```
├── Aplicación/
│   │
│   ├── API/
│   │   │
│   │   ├── Elementos/
│   │   │   │
│   │   │   ├── Recursos/
│   │   │   │   ├── EstilosGraficosPlotly/
│   │   │   │   │   └── EstilosPlotly.py
│   │   │   │   │       └─ Define estilos reutilizables para gráficos Plotly.
│   │   │   │   │
│   │   │   │   ├── ModelosRegresion/
│   │   │   │   │   └── Modelos.py
│   │   │   │   │       └─ Carga los modelos de regresión entrenados.
│   │   │   │   │
│   │   │   │   └── PreparacionDatos/
│   │   │   │       ├── BaseConocimiento/
│   │   │   │       │   └─ Aquí se descargan/colocan los archivos que sirven como base de conocimiento para predicciones.
│   │   │   │       │    
│   │   │   │       ├── CodificadoresCategoricas/
│   │   │   │       │   └─ Aquí se descargan/colocan los archivos que mapean los datos para la predicción de los modelos.
│   │   │   │       │   
│   │   │   │       └── FuncionesPreparacionDatos.py
│   │   │   │           └─ Contiene las funciones necesarias para realizar las predicciones de los modelos.
│   │   │   │   
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
│       └── Interfaz_aplicacion.py
│           └─ Código principal de la interfaz gráfica.
│       
│
├── requirements_aplicación.txt
│    └─ Dependencias necesarias para ejecutar la aplicación.
├── Dockerfile_aplicacion
│    └─ Dockfile para ejecutar la aplicación.
├── Dockerfile_modificacion
│    └─ Dockfile para modificar algunos aspectos de la aplicación(No es estrictamente necesario ya que se puede hacer en local).
└── start.sh
     └─ Comandos que ejecutan diferentes archivos del proyecto para encender la aplicación cuando se usa el Dockerfile_aplicacion
```


### Instalación y ejecución de Aplicación
1. Crear una imagen del contenedor usando Dockerfile_aplicacion (ejemplo: docker build -f .\Dockerfile_aplicacion -t aplicacion_imagen .)
2. Crear un contenedor de la imagen dejando expuesto el puerto indicado en el Dockerfile_aplicacion (ejemplo: docker run --name aplicacion_nombre -p 7861:7860 aplicacion_imagen )
3. Abrir la interfaz de streamlit con el localhost en cualquier buscador dependiendo del apodo que le haya colocado a el puerto. Tambien puede consultar su direccion ip del pc con ipconfig(windows) y abrir la aplicacion en su red local (Ejemplo Localhost: http://localhost:7861) (Ejemplo Red Local: http://192.168.100.5:7861)

## Autor
Manuel Elias Orellana Lavayen 

#!/bin/bash

echo "Descargando Recursos"
python "API/Elementos/Instalando Recursos.py"

echo "Iniciando FastAPI..."

python "API/api_aplicacion.py" &

echo "Esperando a que FastAPI esté disponible..."

until curl -s http://127.0.0.1:8000/docs > /dev/null; do
    sleep 1
done

echo "FastAPI lista."

echo "Iniciando Streamlit..."

streamlit run InterfasGrafica/Interfaz_aplicacion.py --server.address=0.0.0.0 --server.port=7860
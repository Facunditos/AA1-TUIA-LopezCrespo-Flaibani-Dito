# Usar una imagen base de Python
FROM python:3.11-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos necesarios al contenedor
COPY requirements.txt modifiers.py pipeline_model.pkl /app/  

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

COPY inferencia.py /app/inferencia.py

# Definir el comando predeterminado
CMD ["python", "inferencia.py"]
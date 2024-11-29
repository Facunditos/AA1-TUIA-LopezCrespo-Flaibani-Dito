# Usar una imagen base de Python
FROM python:3.11-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos necesarios al contenedor
COPY requirements.txt modelo_rn_best.h5 modifiers.py /app/  

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

#COPY modifiers.py /app/modifiers.py

# Definir el comando predeterminado
ENTRYPOINT ["python", "./modifiers.py"]
CMD ["python", "/app/modifiers.py"]



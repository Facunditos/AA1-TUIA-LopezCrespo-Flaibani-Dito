# Usar una imagen base de Python
FROM python:3.11-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos necesarios al contenedor
COPY requirements.txt pipeline_model.pkl /app/  

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

#inferencia.py rnn.h5 rqst.py 
COPY.. 

# Definir el comando predeterminado
CMD ["python", "inferencia.py"]
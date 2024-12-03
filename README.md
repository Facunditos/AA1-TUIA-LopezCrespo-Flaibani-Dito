## TUIA Aprendizaje Automático I - Trabajo final
### En este proyecto se llevaron adelante tareas vinculadas con el entrenamiento de modelos para predecir variables variables cuantitativas y cualitativas. 
## Comenzar:
- Correr *git clone https://github.com/Facunditos/AA1-TUIA-LopezCrespo-Flaibani-Dito.git* para clonar el repositorio de GitHub en la máquina local.

## Guía para la puesta en producción del modelo de clasificación
En la carpeta Docker se encuentra el Dockerfile con las sentencias para crear la imagen de docker y los archivos que son necesarios para la puesta en producción del modelo. Es necesario alojar en esta carpeta el archivo csv que contendrá los muestras a predecir.
- Correr *docker build -t imagen-rain:v2 ./Docker* para crear la imagen de docker.
- Correr *docker run -v $(pwd):/app/files imagen-rain:v2* desde una terminal de Ubuntu para crear el contenedor donde se ejecturá el programa.
- Luego de ejecutar el comando anterior se alojará en la carpeta del usuario el output con las probabilidades predichas para las muestras.

Las muestras deben corresponderse a ciudades que estén en esta lista.
ciudades admitidas = ['Dartmoor', 'Nuriootpa', 'PerthAirport', 'Uluru', 'Cobar', 'CoffsHarbour', 
               'Walpole', 'Cairns', 'AliceSprings', 'GoldCoast']


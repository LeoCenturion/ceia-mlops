## Machine Learning Operations 1
## CEIA - FIUBA
## 1º Bimestre 2025

## Trabajo Práctico Final

### Grupo

| Autores               | E-mail                    | Nº SIU  |
|---------------------- |---------------------------|---------|
| Leonardo Centurión    | centurionm.leo@gmail.com  | a1803   |
| Braian Desía          | b.desia@hotmail.com       | a1804   |
| Juan José Cardinali   | juanchijc@gmail.com       | a1809   |

# Descripción

El presente servicio proporciona una predicción sobre si va a llover o no en diferentes ciudades de Australia para el día siguiente. El usuario deberá seleccionar la ciudad para que el servicio pueda realizar la prognosis para el día siguiente.

## Fuente

El dataset utilizado para el entrenamiento y testeo del modelo proviene de Kaggle. El mismo se puede descargar de este [link](https://www.kaggle.com/api/v1/datasets/download/jsphyg/weather-dataset-rattle-package).

Los features para nuevas predicciones se toman directamente de la API de [AccuWeather] (https://www.accuweather.com) cuando se utiliza el servicio desde la página web.

# Componentes del proyecto

El presente proyecto involucra los siguientes servicios y herramientas:

1. **Apache Airflow**
   - <ins>Descripción:</ins> Herramienta para programar, monitorear y administrar flujos de trabajo de datos. 
   - <ins>Uso:</ins> Descarga de set desde la fuente, dividir el set en train/test y pre-procesar los datos. Cargar el modelo entrenado a MLflow.

2. **MLflow**: 
   - <ins>Descripción:</ins> Plataforma de código abierto para gestionar el ciclo de vida completo del aprendizaje automático. 
   - <ins>Uso:</ins> Experimentanción y optimización de hiperparámetros. Catalogación de modelos.

3. **MinIO**: 
   - <ins>Descripción:</ins> Servidor de almacenamiento de objetos de alto rendimiento y distribuido. 
   - <ins>Uso:</ins> Almacenamiento del dasaset crudo, dataset pre-procesado por Airflow y artefactos de MLflow.

4. **Flask**:
   - <ins>Descripción:</ins> Framework para servcios HTTP.
   - <ins>Uso:</ins> Disponibilizar el modelo a través de la REST-API y para servir la página web desde donde los usuarios consumen el servicio.

## Diagrama de interacción

![Diagrama](RiA_FlowDiagram.jpg "Componentes del proyecto")

# Instalación

## Pre-requisitos

Para correr el serviciom, asegurarse de tener instalado:
- Docker and Docker Compose

Nota para Windows: Asegúrate de tener Docker Desktop ejecutándose mientras trabajas.

Adicional para levantar y correr Notebooks:
- Python 3.8+
- Poetry


## Detalles de acceso

### 1. Airflow
   - <ins>URL:</ins> http://localhost:8080
   - <ins>Credenciales:</ins>  
     - Username: `airflow`  
     - Password: `airflow`

### 2. MLflow
   - <ins>URL:</ins> http://localhost:5001


### 3. MinIO
   - <ins>URL:</ins> http://localhost:9001
   - <ins>Credenciales:</ins>  
     - Access Key: `minio`
     - Secret Key: `minio123`

### 4. Flask
   - <ins>URL:</ins> http://localhost:5000

---

## Workflow Overview

### Steps in the ETL Pipeline:
1. Data Ingestion: 
   - Downloads the dataset from Kaggle.
   - Stores it in an S3 bucket using MinIO.
2. Feature Engineering:
   - Scales numerical features (`duration`, `tempo`, `loudness`) using `MinMaxScaler`.
   - Retains key features for modeling (`speechiness`, `energy`, `danceability`, `acousticness`).
   - Stores the processed dataset back into MinIO.
3. Dataset Splitting:
   - Splits the dataset into training and testing sets (70/30 split) using stratified sampling.
   - Saves the split datasets in S3.
4. Dataset Registration:
   - Logs dataset metadata and statistics (mean, standard deviation) to S3 and MLflow.

---

## Running the Project

### Step 1: Start the Services
Run the following command to start all services using Docker Compose:

```bash
docker compose --profile all up
```

Pasos para Configurar y Usar PreciosPro AI
Clona este repositorio.

Configuración del entorno (Linux/MacOS):

Si estás en Linux o MacOS, edita el archivo .env y reemplaza AIRFLOW_UID con el UID de tu usuario (puedes encontrarlo con el comando id -u <username>). Esto es necesario para evitar problemas de permisos con Apache Airflow.
Levanta todos los servicios:

En la carpeta raíz de este repositorio, ejecuta el siguiente comando (esto puede llevar unos minutos):
docker compose --profile all up
Verifica que todos los servicios están funcionando:

Usa el comando docker ps -a para asegurarte de que todos los servicios estén en estado "healthy" o revisa en Docker Desktop.
Accede a los servicios disponibles:

Apache Airflow: http://localhost:8080(Usuario: airflow, Password: airflow)
MLflow: http://localhost:5005
MinIO (administración de buckets): http://localhost:9001(Usuario: minio, Password: minio123)
Streamlit: http://localhost:8501/
(Opcional) Ejecución de ETL en Airflow:

En Apache Airflow, ejecuta el ETL haciendo clic en el botón de "play". Espera unos minutos hasta que se complete.
(Opcional) Visualiza los archivos en MinIO:

Ahora podrás visualizar en MinIO el bucket con los archivos que se utilizarán en el entrenamiento del modelo.
Entrenamiento del modelo:

Ejecuta el notebook entero dentro de la carpeta ./notebooks para realizar el entrenamiento del modelo. Si no realizaste los puntos 6 y 7, desde el notebook podes ejecutar el ETL en airflow (primera celda de código).
Visualización de resultados:

Podrás visualizar en MLflow el modelo entrenado, junto con sus métricas más importantes, así como en MinIO.
Predicción con tu vivienda:

¡Ya casi estás! Ahora entra en la API, llena los datos de tu inmueble, y haz clic en "Enviar".
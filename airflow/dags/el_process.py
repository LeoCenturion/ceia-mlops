# ----------------------------------------------------------------

import datetime

# Airflow utilities
from airflow.decorators import dag, task

# ----------------------------------------------------------------

# Set default dag parameters.
default_args = {
    'owner': 'Braian, Leo & Juan',
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}

md_text = """
### EL Process for rain_in_australia
"""

@dag(
    dag_id="el_rain_in_australia",
    description="EL process for rain_in_australia using TaskFlow, getting data from source",
    doc_md=md_text,
    tags=["EL", "rain_in_australia"],
    default_args=default_args,
    catchup=False,
    schedule_interval=None,
)

def process_el_ria():

    @task.virtualenv(
        task_id="get_rawdata",
        requirements=["awswrangler==3.6.0"],
    )
    def get_rawdata(url, filename):
        """
        Load the raw data from source into S3 bucket
        """
        from utilities import download_data
        import logging

        logger = logging.getLogger("airflow.task")

        # Llama a la función de descarga y obtiene el estado
        status = download_data(url, filename)
        for message in status:
            logger.info(message)

    # URLs y nombres de los archivos
    # Base de datos principal
    url1 = 'https://www.kaggle.com/api/v1/datasets/download/jsphyg/weather-dataset-rattle-package'
    filename1 = 'rains'

    # Base de datos auxiliar con coordenadas geográficas
    url2 = 'https://simplemaps.com/static/data/world-cities/basic/simplemaps_worldcities_basicv1.77.zip'
    filename2 = 'simplemaps_worldcities_basicv1.77.zip'

    # Workflow
    # Llamar a get_rawdata para cada URL y nombre de archivo
    get_rawdata(url1, filename1) >> get_rawdata(url2, filename2)

# Inicia el DAG
dag = process_el_ria()
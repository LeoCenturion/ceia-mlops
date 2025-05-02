
# ----------------------------------------------------------------

import datetime
import awswrangler as wr

import sys
import os

# Agregar el directorio de la librería a sys.path
lib_path = os.path.abspath(r'..\..\service\src')
sys.path.append(lib_path)

# Airflow utilities
from airflow.decorators import dag, task

# ----------------------------------------------------------------


# Some general parameters
folder_path = "s3://data/"     # Directory where data is stored


# Set default dag parameters.
default_args = {
    'owner': 'Braian, Leo & Juan',
    # 'email': ['xx@gmail.com'],
    # 'email_on_failure': False,
    # 'email_on_retry': False,
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}

md_text = """
### TL Process for rain_in_australia
"""

@dag(
    dag_id="tl_rain_in_australia",
    description="TL process for rain_in_australia using TaskFlow, separating the dataset into training and testing sets",
    doc_md=md_text,
    tags=["TL", "rain_in_australia"],
    default_args=default_args,
    catchup=False,
    schedule_interval=None,
    # start_date=days_ago(2),
)


    
def process_tl_ria():
    
    @task.virtualenv(
        task_id="split_dataset",
        requirements=["pandas~=1.5",
                       "scikit-learn==1.3.2",
                        "awswrangler==3.6.0",
                    ],
    )

    def split_dataset():
        """
        Genera el dataset y obtiene set de testeo y evaluación
        """
        import logging
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split

        logger = logging.getLogger("airflow.task")

        # Load the dataset from S3
        s3_input_path = folder_path+"weatherAUS.csv"  # Cambia esta ruta según corresponda
        logger.info("Reading dataset from: %s", s3_input_path)
        rain_df = wr.s3.read_csv(s3_input_path)  # Leer desde S3 usando awswrangler
        logger.info("Dataset reading successfully finished")

        # Preprocess dataset
        logger.info("Getting features and label from dataset")
        X_full = rain_df.drop(columns=['RainTomorrow'])  # Drop the target column from features
        y_full = np.where(rain_df['RainTomorrow'] == "Yes", 1, 0)  # Target variable

        # Split the dataset
        logger.info("Splitting dataset")
        ftest = 0.20  # Data fraction for "training"
        logger.info("Test size: %s", ftest)
        logger.info("Stratify: True")

        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, 
            test_size=ftest,        
            stratify=y_full,  # Keep class proportions the same       
        )

        logger.info("X_train dimension: %s", X_train.shape)
        logger.info("y_train dimension: %s", y_train.size)
        logger.info("X_test dimension: %s", X_test.shape)
        logger.info("y_test dimension: %s", len(y_test))

        # Saving datasets into S3
        output_prefix = folder_path + "datasets/"  # Cambiar esta ruta según sea necesario
        logger.info("Saving datasets into: %s", output_prefix)
        wr.s3.to_csv(df=X_train, path=f"{output_prefix}X_train.csv", index=False)
        wr.s3.to_csv(df=y_train, path=f"{output_prefix}y_train.csv", index=False)
        wr.s3.to_csv(df=X_test, path=f"{output_prefix}X_test.csv", index=False)
        wr.s3.to_csv(df=y_test, path=f"{output_prefix}y_test.csv", index=False)
        
        logger.info("Dataset splitting successfully finished")
    
    def prepo_pipeline():
        """
        Aplica pipeline de preprocesamiento de datos
        """
        
        import logging
        import pandas as pd
        from pipeline import prepro_pipeline
        
        logger = logging.getLogger("airflow.task")
        
        X_train_path = folder_path + 'X_train.csv'
        X_test_path = folder_path + 'X_test.csv'
        
        logger.info("Reading train and test dataset from : %s", folder_path)
        X_train = pd.read_csv(X_train_path)
        X_test = pd.read_csv(X_test_path)
        logger.info("Train and Test Datasets reading successfully finished")        

        logger.info('Applying pre-processing pipeline: Fit and transform Train data')
        X_train_transformed = prepro_pipeline.fit_transform(X_train)

        logger.info('Applying pre-processing pipeline: Transforming test data based on training fit')
        X_test_transformed = prepro_pipeline.transform(X_test)

        # Saving datasets into S3
        output_prefix = folder_path + "datasets/"  # Cambiar esta ruta según sea necesario
        logger.info("Saving transformed datasets into: %s", folder_path)
        wr.s3.to_csv(df=X_train_transformed, path=f"{output_prefix}X_train_transformed.csv", index=False)
        wr.s3.to_csv(df=X_test_transformed, path=f"{output_prefix}X_test_transformed.csv", index=False)
        logger.info("Dataset features transform successfully finished")

    # Workflow    
    split_dataset() >> prepo_pipeline
           
dag = process_tl_ria()
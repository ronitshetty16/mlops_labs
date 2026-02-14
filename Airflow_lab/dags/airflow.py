from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from src.lab import (
    load_data, 
    data_preprocessing, 
    build_save_model, 
    evaluate_model
)

# Define default arguments for your DAG
default_args = {
    'owner': 'Ronit',  # Your name
    'start_date': datetime(2026, 2, 14),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'Wine_Quality_Prediction',
    default_args=default_args,
    description='Random Forest Regressor for wine quality prediction',
    catchup=False,
) as dag:
    
   
    load_data_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
    )
    
    
    preprocessing_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
    )
    
    
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=build_save_model,
        op_args=[preprocessing_task.output, "rf_model.pkl"],
    )
    
    
    evaluate_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        op_args=["rf_model.pkl", train_model_task.output],
    )
    
    
    load_data_task >> preprocessing_task >> train_model_task >> evaluate_task

if __name__ == "__main__":
    dag.test()
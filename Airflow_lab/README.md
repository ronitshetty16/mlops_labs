# Wine Quality Prediction - Airflow Pipeline

## Overview:-
An Airflow DAG that trains a Random Forest Regressor to predict wine quality scores (0-10) based on chemical properties.

## Pipeline Tasks

1. **load_data** - Load wine dataset from sklearn
2. **data_preprocessing** - Split data and scale features using StandardScaler
3. **build_save_model** - Train Random Forest Regressor and save model
4. **evaluate_model** - Evaluate on test data and report R² score
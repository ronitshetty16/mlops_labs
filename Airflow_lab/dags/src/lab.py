import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import base64

def load_data():
    """Loads wine dataset."""
    print("Loading wine quality dataset...")
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['quality'] = wine.target
    
    serialized_data = pickle.dumps(df)
    return base64.b64encode(serialized_data).decode("ascii")

def data_preprocessing(data_b64: str):
    """Preprocesses and splits data."""
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)
    
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    data_dict = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train.values,
        'y_test': y_test.values
    }
    
    serialized_data = pickle.dumps(data_dict)
    return base64.b64encode(serialized_data).decode("ascii")

def build_save_model(data_b64: str, filename: str):
    """Trains Random Forest Regressor."""
    data_bytes = base64.b64decode(data_b64)
    data_dict = pickle.loads(data_bytes)
    
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    print(f"Training R² score: {train_score:.4f}")
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    
    return data_b64

def evaluate_model(filename: str, data_b64: str):
    """Evaluates model on test data."""
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    model = pickle.load(open(output_path, "rb"))
    
    data_bytes = base64.b64decode(data_b64)
    data_dict = pickle.loads(data_bytes)
    
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"\n{'='*50}")
    print(f"Test R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"{'='*50}")
    print(f"\nSample predictions vs actual:")
    for i in range(5):
        print(f"  Predicted: {predictions[i]:.2f} | Actual: {y_test[i]}")
    print(f"{'='*50}\n")
    
    return float(r2)
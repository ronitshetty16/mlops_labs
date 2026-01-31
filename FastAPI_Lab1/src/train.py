from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from data import load_data, split_data

def fit_model(X_train, y_train):
    """
    Train a Random Forest Regressor and save the model to a file.
    
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)
    joblib.dump(rf_regressor, "../model/house_model.pkl")
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train model
    fit_model(X_train, y_train)
    
    # Optional: Evaluate on test set
    model = joblib.load("../model/house_model.pkl")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R2 Score: {r2:.4f}")
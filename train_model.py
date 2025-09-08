import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap

def train_and_evaluate_model(X, y):
    """
    Splits data, trains an XGBoost model, and evaluates its performance.
    """
    # Split data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the XGBoost Regressor model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)
    
    # Evaluate model performance using key regression metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("Model Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")
    
    return model, X_test, y_test

def explain_prediction_shap(model, X_test, query_instance):
    """
    Generates a SHAP explanation for a single prediction.
    """
    # Create a SHAP explainer for the XGBoost model
    explainer = shap.Explainer(model)
    
    # Calculate SHAP values for the query instance
    shap_values = explainer(query_instance)
    
    return shap_values
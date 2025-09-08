# main.py
import pandas as pd
import os
import shap
from train_model import train_and_evaluate_model, explain_prediction_shap
from data_preprocessing import prepare_data, load_data

if __name__ == "__main__":
    # Define the file path for the preprocessed dataset
    preprocessed_file_path = "Dataset.csv"

    # --- Step 1: Load the preprocessed and saved dataset ---
    print("Step 1: Loading the preprocessed dataset...")
    
    data = load_data(preprocessed_file_path)
    
    if data is None:
        exit()

    # --- Step 2: Prepare the data for the model ---
    X, y = prepare_data(data)
    
    print("Data preparation complete. Ready to train the model.\n")

    # --- Step 3: Train and evaluate the model ---
    print("Step 2: Training and evaluating the XGBoost model...")
    model, X_test, y_test = train_and_evaluate_model(X, y)

    # --- Step 4: Provide a SHAP explanation ---
    print("\nStep 3: Generating a SHAP explanation...")
    
    # Select a driver's data for explanation (e.g., the first driver in the test set)
    driver_instance = X_test.iloc[[0]]
    original_score = model.predict(driver_instance)[0]
    
    print(f"\nOriginal Nova Score for driver: {original_score:.2f}")
    print(f"Driver's features:\n{driver_instance.iloc[0]}\n")
    
    # Get SHAP values for the prediction
    shap_values = explain_prediction_shap(model, X, driver_instance)

    # Visualize the SHAP explanation
    shap.plots.waterfall(shap_values[0])
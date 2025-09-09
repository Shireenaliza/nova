import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Configuration ---
DATA_PATH = 'Dataset.csv'
MODEL_PATH = 'xgb_model.joblib'
SCALER_PATH = 'scalers.joblib'
TARGET_COLUMN = 'nova_score' # A new column we'll create for the target variable

# --- Feature Engineering Weights (for Combined Performance Score) ---
# You can adjust these weights based on business logic. They should sum to 1.
W1 = 0.3  # normalized_avg_tip_percentage
W2 = 0.4  # normalized_avg_customer_rating
W3 = 0.3  # normalized_trip_consistency

def load_data(path):
    """Loads the dataset and performs initial data cleaning."""
    df = pd.read_csv(path)
    # Convert 'age' and 'gender' to numeric if they are not already.
    # Note: 'gender' can be one-hot encoded for other models, but for this project, we exclude it.
    return df

def create_target_and_features(df):
    """
    Creates the target variable (Nova Score) and engineered features.
    The Nova Score is a proxy for creditworthiness, based on a weighted sum of
    normalized performance metrics.
    """
    # Create the 'nova_score' target variable. This is a synthetic target
    # for a regression problem, based on a combination of positive indicators.
    df['normalized_avg_tip_percentage'] = MinMaxScaler().fit_transform(df[['avg_tip_percentage']])
    df['normalized_rating'] = MinMaxScaler().fit_transform(df[['rating']])
    df['normalized_total_trips'] = MinMaxScaler().fit_transform(df[['total_trips']])

    df[TARGET_COLUMN] = (
        W1 * df['normalized_avg_tip_percentage'] +
        W2 * df['normalized_rating'] +
        W3 * df['normalized_total_trips']
    ) * 100 # Scale to 0-100 for a score

    # Drop intermediate columns
    df.drop(columns=['normalized_avg_tip_percentage', 'normalized_rating', 'normalized_total_trips'], inplace=True)
    return df

def feature_engineer(df):
    """
    Creates the engineered features based on the provided formulae.
    Handles edge cases as described.
    """
    # Epsilon for numerical stability
    epsilon = 1e-6

    # Income Stability Index
    df['Income_Stability_Index'] = df['std_weekly_income'] / (df['avg_weekly_income'] + epsilon)
    # Clip large values to prevent outliers from skewing the data
    df['Income_Stability_Index'] = df['Income_Stability_Index'].clip(upper=10)

    # Fare-to-Tip Ratio
    df['Fare-to-Tip_Ratio'] = df['avg_tip_percentage'] / (df['avg_total_fare'] + epsilon)
    df['Fare-to-Tip_Ratio'] = df['Fare-to-Tip_Ratio'].clip(upper=10)

    # Trip Consistency
    df['Trip_Consistency'] = df['number_of_weeks_on_platform'] / (df['total_trips'] + epsilon)

    return df

def train_model(X_train, y_train):
    """Trains the XGBoost model."""
    xgb_regressor = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42
    )
    xgb_regressor.fit(X_train, y_train)
    return xgb_regressor

# def main():
#     """Main function to run the entire training pipeline."""
#     print("🎬 Starting model training pipeline...")

#     # 1. Load Data
#     df = load_data(DATA_PATH)
#     print("✅ Data loaded successfully.")

#     # 2. Create Target Variable (Nova Score)
#     df = create_target_and_features(df)
#     print("✅ Target 'Nova Score' created.")

#     # 3. Feature Engineering
#     df = feature_engineer(df)
#     print("✅ Engineered features created.")

#     # 4. Prepare Data for Model
#     # Features to be used for the model
#     features = [
#         'avg_total_fare', 'std_total_fare', 'avg_tip_percentage',
#         'total_trips', 'avg_trip_distance', 'avg_weekly_income',
#         'std_weekly_income', 'number_of_weeks_on_platform', 'rating',
#         'Income_Stability_Index', 'Fare-to-Tip_Ratio', 'Trip_Consistency'
#     ]

#     # Drop age and gender to avoid bias, as per the prompt
#     X = df[features]
#     y = df[TARGET_COLUMN]

#     # Data scaling: Crucial for many ML models
#     scaler_X = MinMaxScaler()
#     X_scaled = scaler_X.fit_transform(X)

#     # 5. Split Data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_scaled, y, test_size=0.2, random_state=42
#     )
#     print(f"✅ Data split: Training set size = {len(X_train)}, Test set size = {len(X_test)}")

#     # 6. Train the Model
#     model = train_model(X_train, y_train)
#     print("✅ XGBoost model training complete.")

#     # 7. Evaluate the Model
#     y_pred = model.predict(X_test)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     print(f"📈 Model Evaluation - RMSE on test set: {rmse:.2f}")

#     # 8. Save Model and Scaler
#     joblib.dump(model, MODEL_PATH)
#     joblib.dump(scaler_X, 'scaler_X.joblib')
#     # Store the list of features for consistent input to the app
#     joblib.dump(features, 'model_features.joblib')

#     print(f"💾 Model saved to '{MODEL_PATH}'")
#     print(f"💾 Scaler saved to 'scaler_X.joblib'")
#     print(f"💾 Feature list saved to 'model_features.joblib'")
#     print("🚀 Training pipeline finished. You can now run the Streamlit app.")

def main():
    """Main function to run the entire training pipeline."""
    print("🎬 Starting model training pipeline...")

    # 1. Load Data
    df = load_data(DATA_PATH)
    print("✅ Data loaded successfully.")

    # 2. Create Target Variable (Nova Score)
    df = create_target_and_features(df)
    print("✅ Target 'Nova Score' created.")

    # 3. Feature Engineering
    df = feature_engineer(df)
    print("✅ Engineered features created.")

    # --- NEW ADDITION ---
    # 4. Save the processed DataFrame to a new CSV file
    processed_data_path = 'processed_dataset.csv'
    df.to_csv(processed_data_path, index=False)
    print(f"✅ Processed data with new columns saved to '{processed_data_path}'.")
    # --- END OF NEW ADDITION ---

    # 5. Prepare Data for Model
    # Features to be used for the model
    features = [
        'avg_total_fare', 'std_total_fare', 'avg_tip_percentage',
        'total_trips', 'avg_trip_distance', 'avg_weekly_income',
        'std_weekly_income', 'number_of_weeks_on_platform', 'rating',
        'Income_Stability_Index', 'Fare-to-Tip_Ratio', 'Trip_Consistency'
    ]

    # Drop age and gender to avoid bias, as per the prompt
    X = df[features]
    y = df[TARGET_COLUMN]

    # Data scaling: Crucial for many ML models
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    # 6. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"✅ Data split: Training set size = {len(X_train)}, Test set size = {len(X_test)}")

    # 7. Train the Model
    model = train_model(X_train, y_train)
    print("✅ XGBoost model training complete.")

    # 8. Evaluate the Model with multiple metrics
    y_pred = model.predict(X_test)

    # Calculate and print evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"📈 Model Evaluation:")
    print(f"  - Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"  - Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  - R-squared (R²): {r2:.2f}")
    
    # 9. Save Model and Scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler_X, 'scaler_X.joblib')
    # Store the list of features for consistent input to the app
    joblib.dump(features, 'model_features.joblib')

    print(f"💾 Model saved to '{MODEL_PATH}'")
    print(f"💾 Scaler saved to 'scaler_X.joblib'")
    print(f"💾 Feature list saved to 'model_features.joblib'")
    print("🚀 Training pipeline finished. You can now run the Streamlit app.")

if __name__ == "__main__":
    main()
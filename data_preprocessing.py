import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """
    Loads the dataset from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

def engineer_features(df):
    """
    Engineers new features from the raw data.
    """
    # Create the Income Stability Index, handling division by zero
    df['income_stability_index'] = df['avg_weekly_income'] / (df['std_weekly_income'] + 1e-6)

    # Create the Fare-to-Tip Ratio, handling division by zero
    df['fare_to_tip_ratio'] = df['avg_total_fare'] / (df['avg_tip_percentage'] + 1e-6)

    # Create the Trip Consistency feature
    df['trip_consistency'] = df['total_trips'] / (df['number_of_weeks_on_platform'] + 1e-6)

    # Normalize features for the Combined Performance Score
    scaler = MinMaxScaler()
    df[['avg_tip_percentage_normalized', 'rating_normalized']] = scaler.fit_transform(
        df[['avg_tip_percentage', 'rating']]
    )

    # Create the Combined Performance Score (Example weights)
    w1, w2 = 0.6, 0.4
    df['combined_performance_score'] = (w1 * df['avg_tip_percentage_normalized']) + (w2 * df['rating_normalized'])

    return df

def prepare_data(df, target_column='nova_score'):
    """
    Prepares features (X) and target label (y).
    """
    # Features to be used for model training
    features = [
        'avg_total_fare', 'std_total_fare', 'avg_tip_percentage', 'total_trips',
        'avg_trip_distance', 'avg_weekly_income', 'std_weekly_income',
        'number_of_weeks_on_platform', 'rating', 'income_stability_index',
        'fare_to_tip_ratio', 'trip_consistency',
    ]
    
    # Drop rows with any NaN or infinite values in the selected features
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features + [target_column])

    X = df[features]
    y = df[target_column]
    
    return X, y
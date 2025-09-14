import pandas as pd
import joblib
import sys
import os    
import numpy as np

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (the project root)
project_root = os.path.dirname(script_dir)
# Add the project root to the system path to allow importing modules
sys.path.append(project_root)

# Import the data preprocessing function from the src directory
from src.data_preprocessing import load_and_preprocess

def feature_engineering(data):

    data['failed_login_rate'] = data['failed_logins'] / (data['login_attempts'] + 1e-6)
    data['session_speed'] = data['session_duration'] / (data['network_packet_size'] + 1e-6)
    data['protocol_encryption_combo'] = data['protocol_type'] + '_' + data['encryption_used'].astype(str)
    
    return data

if __name__ == "__main__":
    # Define file paths
    data_file_path = os.path.join(project_root, 'data', 'cybersecurity_intrusion_data.csv')
    model_dir = os.path.join(project_root, 'models')
    
    # Load the trained model and preprocessor
    try:
        # Loading the specific model file ids_model.pkl
        rf_model = joblib.load(os.path.join(model_dir, 'ids_model.pkl'))
        preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.pkl'))
        print("Model and preprocessor loaded successfully.")
    except FileNotFoundError:
        print("Error: Model or preprocessor files not found. Please run src/train_random_forest.py first.")
        sys.exit(1)

    # Load the entire dataset to get a sample data point for prediction
    data = pd.read_csv(data_file_path)
    
    # Separate features and target
    X_raw = data.drop('attack_detected', axis=1)
    y_raw = data['attack_detected']
    
    # We will pick the 3672th data point as our example
    sample_index = 3672
    sample_data = X_raw.iloc[sample_index:sample_index+1].copy()
    actual_value = y_raw.iloc[sample_index]

    print(f"\n--- Predicting on a Single Data Point (Index: {sample_index}) ---")
    print("\nRaw data from the CSV file:")
    print(sample_data.to_string())
    
    # Apply feature engineering to the sample data
    engineered_sample_data = feature_engineering(sample_data)

    # Preprocess the sample data using the loaded preprocessor
    processed_sample = preprocessor.transform(engineered_sample_data)
    
    # Make a prediction using the loaded model
    prediction = rf_model.predict(processed_sample)
    
    # Print the results
    print(f"\nActual value: {actual_value}")
    print(f"Model prediction: {prediction[0]}")

    if prediction[0] == actual_value:
        print("\nPrediction matches the actual value. The model is working as expected!")
    else:
        print("\nPrediction does NOT match the actual value. Further investigation may be needed.")


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def feature_engineering(data):
    data['failed_login_rate'] = data['failed_logins'] / (data['login_attempts'] + 1e-6)
    data['session_speed'] = data['session_duration'] / (data['network_packet_size'] + 1e-6)
    data['protocol_encryption_combo'] = data['protocol_type'] + '_' + data['encryption_used']
    
    return data

def load_and_preprocess(filepath):
    data = pd.read_csv(filepath)

    data = feature_engineering(data)

    X = data.drop('attack_detected', axis=1)
    y = data['attack_detected']

    # Identify categorical and numerical columns, including the new ones
    numerical_cols = ['network_packet_size', 'login_attempts', 'session_duration', 
                      'ip_reputation_score', 'failed_logins', 'unusual_time_access',
                      'failed_login_rate', 'session_speed']
    
    categorical_cols = ['protocol_type', 'encryption_used', 'browser_type', 
                        'protocol_encryption_combo']

    # Handle outliers in numerical columns using IQR capping
    for col in numerical_cols:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X[col] = np.clip(X[col], lower_bound, upper_bound)

    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='drop'
    )

    # Apply the preprocessing to the features
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor, numerical_cols, categorical_cols

if __name__ == "__main__":
    file_path = '../data/cybersecurity_intrusion_data.csv'
    
    # Call the function and get the preprocessed data and the lists of columns
    X_processed, y, preprocessor, numerical_cols, categorical_cols = load_and_preprocess(file_path)

    # Convert the processed data back to a DataFrame to inspect it
    one_hot_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    all_cols = numerical_cols + list(one_hot_cols)

    X_df = pd.DataFrame(X_processed, columns=all_cols)

    print("Shape of preprocessed features (X):", X_df.shape)
    print("Shape of target variable (y):", y.shape)

    print("\nFirst 5 rows of the preprocessed DataFrame:")
    print(X_df.head())
    
    print("\nData types of preprocessed DataFrame:")
    print(X_df.dtypes)

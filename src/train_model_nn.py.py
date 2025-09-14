import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from imblearn.over_sampling import SMOTE

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (the project root)
project_root = os.path.dirname(script_dir)
# Add the project root to the system path to allow importing modules
sys.path.append(project_root)

# Import the data preprocessing function from the src directory
from src.data_preprocessing import load_and_preprocess

if __name__ == "__main__":
    # Define the file path for the dataset
    file_path = os.path.join(project_root, 'data', 'cybersecurity_intrusion_data.csv')
    
    # Load and preprocess data using the function from the data_preprocessing script
    X, y, preprocessor, numerical_cols, categorical_cols = load_and_preprocess(file_path)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- SMOTE section is now enabled ---
    print("Training the model on the balanced data using SMOTE.")
    print("Original training data class distribution:")
    unique_classes, class_counts = pd.Series(y_train).value_counts().sort_index().index, pd.Series(y_train).value_counts().sort_index().values
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Class {cls}: {count} samples")
    
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print("\nBalanced training data class distribution (after SMOTE):")
    unique_classes, class_counts = pd.Series(y_train_balanced).value_counts().sort_index().index, pd.Series(y_train_balanced).value_counts().sort_index().values
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Class {cls}: {count} samples")

    # Build the Neural Network Model with more layers ---
    # Get the number of features (columns) from the preprocessed data
    n_features = X_train_balanced.shape[1]

    # Initialize a sequential model
    model = Sequential([
        # Input layer and first hidden layer with 64 neurons
        Dense(64, activation='relu', input_shape=(n_features,)),
        # Second hidden layer with 32 neurons
        Dense(32, activation='relu'),
        # Third hidden layer with 16 neurons
        Dense(16, activation='relu'),
        # Output layer with 1 neuron and sigmoid activation for binary classification
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Print a summary of the model architecture
    print("\n--- Neural Network Model Summary ---")
    model.summary()
    print("\nTraining the Neural Network model...")

    # Train the model on the balanced data
    history = model.fit(X_train_balanced, y_train_balanced, 
                        epochs=25, 
                        batch_size=32, 
                        validation_split=0.1,  # Use 10% of the training data for validation
                        verbose=1)
    print("Model training complete.")

    # Make predictions on the UNSEEN test data
    # Predictions are probabilities, so we convert them to class labels (0 or 1)
    predictions = (model.predict(X_test) > 0.5).astype("int32")

    # Evaluate the model's performance
    print("\n--- Model Evaluation on Test Data ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Save the trained model to the models folder
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    # Save the Keras model
    model.save(os.path.join(models_dir, 'ids_nn_model.h5'))
    # Save the preprocessor
    import joblib
    joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor.pkl'))
    print(f"\nModel and preprocessor saved to '{models_dir}'")

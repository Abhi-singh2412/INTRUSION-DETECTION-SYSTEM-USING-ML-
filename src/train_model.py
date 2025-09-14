import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import os
import joblib
import matplotlib.pyplot as plt 
import seaborn as sns 
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
    # Define the file path for the dataset using a portable method
    file_path = os.path.join(project_root, 'data', 'cybersecurity_intrusion_data.csv')
    
    # Load and preprocess data using the function from the data_preprocessing script
    X, y, preprocessor, numerical_cols, categorical_cols = load_and_preprocess(file_path)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Note: Random Forest models are often robust to imbalanced data and may not
    # require SMOTE. For this experiment, we will not use it.
    print("--- Training a Random Forest Model ---")
    print("Training on the original, imbalanced data.")
    print("Training data class distribution:")
    unique_classes, class_counts = pd.Series(y_train).value_counts().sort_index().index, pd.Series(y_train).value_counts().sort_index().values
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Class {cls}: {count} samples")

    # Initialize and train the Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=3000, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    print("\nModel training complete.")

    # Make predictions on the UNSEEN test data
    predictions = rf_model.predict(X_test)

    # Evaluate the model's performance
    print("\n--- Model Evaluation on Test Data ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Detected (0)', 'Detected (1)'],
                yticklabels=['Not Detected (0)', 'Detected (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Random Forest Confusion Matrix')
    plt.show()

    # Save the trained model to the models folder
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(rf_model, os.path.join(models_dir, 'ids_random_forest_model.pkl'))
    joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor.pkl'))
    print(f"\nModel and preprocessor saved to '{models_dir}'")

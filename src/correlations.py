import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Add the src directory to the system path to import modules
sys.path.append('.')
from src.data_preprocessing import load_and_preprocess

# Run the preprocessing function to get the data
file_path = 'data/cybersecurity_intrusion_data.csv'
X_processed, y, preprocessor, numerical_cols, categorical_cols = load_and_preprocess(file_path)

# Convert the processed data back to a DataFrame to inspect it
one_hot_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_cols = numerical_cols + list(one_hot_cols)
X_df = pd.DataFrame(X_processed, columns=all_cols)


corr_matrix = X_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Features')
plt.show()

X_df['attack_detected'] = y

target_corr = X_df.corrwith(X_df['attack_detected']).sort_values(ascending=False)
print("\nCorrelation with the 'attack_detected' target variable:")
print(target_corr)

X_df = X_df.drop('attack_detected', axis=1)
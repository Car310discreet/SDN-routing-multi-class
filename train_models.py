import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def train_and_save_models():
    # Make sure output directory exists
    os.makedirs('models', exist_ok=True)

    csv_path = 'experiment/result_data.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # First 19 columns are features, 20th column is target traffic_type_numeric
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Train Logistic Regression Model
    print("Training Logistic Regression model...")
    lr = LogisticRegression(solver='liblinear', random_state=42)
    lr.fit(X, y)

    # Save Logistic Regression model
    joblib.dump(lr, 'models/logistic_regression_model.pkl')
    print("Logistic Regression model saved to models/logistic_regression_model.pkl")

    # Train XGBoost Model
    print("Training XGBoost model...")
    # Map labels to 0-indexed for XGBoost if necessary
    unique_labels = sorted(np.unique(y))
    label_map = {val: idx for idx, val in enumerate(unique_labels)}
    y_xgb = np.array([label_map[v] for v in y])

    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X, y_xgb)

    # Save XGBoost model
    xgb_model.save_model('models/xgboost_binary_classification.json')
    print("XGBoost model saved to models/xgboost_binary_classification.json")

    print("\nAll models trained and saved successfully!")

if __name__ == '__main__':
    train_and_save_models()

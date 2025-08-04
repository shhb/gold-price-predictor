# train.py

import os
import pandas as pd
import boto3
import joblib
from xgboost import XGBRegressor

# === Configuration ===
BUCKET_NAME = "gold-price-predict-sagemaker-ai"
OBJECT_KEY = "xauusd1440v1.csv"
LOCAL_DATA_PATH = "data/gold.csv"
MODEL_DIR = "model"
MODEL_NAME = "gold_xgb_model.joblib"

def download_from_s3(bucket: str, key: str, local_path: str):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"⬇️ Downloading {key} from S3 bucket: {bucket}")
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
    print("✅ Download complete!")

def load_and_preprocess_data(path: str):
    df = pd.read_csv(path)

    # Combine Date and Time if present
    if "Date" in df.columns and "Time" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    elif "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    else:
        raise ValueError("CSV must contain either ['Date', 'Time'] or 'Timestamp' columns")

    df = df.sort_values("Timestamp")

    # Create prediction target: next close price
    df["Target"] = df["Close"].shift(-1)
    df = df.dropna()

    X = df[["Open", "High", "Low", "Close", "Volume"]]
    y = df["Target"]

    return X, y

def train_model(X_train, y_train):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, model_dir, model_filename):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, model_filename)
    joblib.dump(model, path)
    print(f"💾 Model saved to: {path}")

def main():
    download_from_s3(BUCKET_NAME, OBJECT_KEY, LOCAL_DATA_PATH)

    print("📊 Loading and preprocessing data...")
    X, y = load_and_preprocess_data(LOCAL_DATA_PATH)

    # Use last 30 samples for "test-like" split
    print("🔀 Splitting data...")
    X_train = X[:-30]
    y_train = y[:-30]

    print(f"📦 Training data shape: {X_train.shape}")

    print("🧠 Training XGBoost model...")
    model = train_model(X_train, y_train)

    print("💾 Saving model...")
    save_model(model, MODEL_DIR, MODEL_NAME)

    print("✅ Training completed successfully!")

if __name__ == "__main__":
    main()


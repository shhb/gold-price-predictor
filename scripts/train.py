# train.py

import os
import pandas as pd
import boto3
import joblib
from xgboost import XGBRegressor

from sagemaker import Session
from sagemaker.model import ModelPackage
from botocore.exceptions import ClientError

# === Configuration ===
BUCKET_NAME = "gold-price-predict-sagemaker-ai"
OBJECT_KEY = "xauusd1440v1.csv"
LOCAL_DATA_PATH = "data/gold.csv"
MODEL_DIR = "model"
MODEL_NAME = "gold_xgb_model.joblib"
S3_MODEL_KEY = f"models/{MODEL_NAME}"
MODEL_PACKAGE_GROUP = "GoldPriceXGBGroup"

# === Functions ===

def download_from_s3(bucket: str, key: str, local_path: str):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"⬇️ Downloading {key} from S3 bucket: {bucket}")
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
    print("✅ Download complete!")

def load_and_preprocess_data(path: str):
    df = pd.read_csv(path)

    if "Date" in df.columns and "Time" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    elif "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    else:
        raise ValueError("CSV must contain ['Date' & 'Time'] or 'Timestamp' columns")

    df.sort_values("Timestamp", inplace=True)
    df["Target"] = df["Close"].shift(-1)
    df.dropna(inplace=True)

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
    return path

def upload_model_to_s3(local_path: str, bucket: str, s3_key: str):
    print(f"📤 Uploading model to s3://{bucket}/{s3_key}...")
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, s3_key)
    print("✅ Upload complete!")
    return f"s3://{bucket}/{s3_key}"

def register_model(model_s3_uri: str, model_package_group: str):
    print("🧾 Registering model to SageMaker Model Registry...")
    sagemaker_session = Session()
    role_arn = os.environ.get("SAGEMAKER_ROLE_ARN")  # Must be passed from CodeBuild environment

    if not role_arn:
        raise RuntimeError("Missing SAGEMAKER_ROLE_ARN environment variable")

    model_package = sagemaker_session.create_model_package_from_containers(
        model_package_group_name=model_package_group,
        inference_specification={
            "containers": [{
                "image": "683313688378.dkr.ecr.us-east-2.amazonaws.com/sagemaker-xgboost:1.5-1",
                "model_data_url": model_s3_uri,
            }],
            "supported_content_types": ["text/csv"],
            "supported_response_mime_types": ["text/csv"],
        },
        model_approval_status="PendingManualApproval",
        role_arn=role_arn,
    )
    print(f"✅ Model registered: {model_package}")


# === Main ===

def main():
    download_from_s3(BUCKET_NAME, OBJECT_KEY, LOCAL_DATA_PATH)

    print("📊 Loading and preprocessing data...")
    X, y = load_and_preprocess_data(LOCAL_DATA_PATH)

    print("🔀 Splitting data...")
    X_train = X[:-30]
    y_train = y[:-30]
    print(f"📦 Training data shape: {X_train.shape}")

    print("🧠 Training XGBoost model...")
    model = train_model(X_train, y_train)

    print("💾 Saving model...")
    local_model_path = save_model(model, MODEL_DIR, MODEL_NAME)

    print("☁️ Uploading model to S3...")
    model_s3_uri = upload_model_to_s3(local_model_path, BUCKET_NAME, S3_MODEL_KEY)

    print("📜 Registering model in SageMaker Registry...")
    register_model(model_s3_uri, MODEL_PACKAGE_GROUP)

    print("🎉 All done!")

if __name__ == "__main__":
    main()

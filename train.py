import os
from io import StringIO
import boto3
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. AWS S3 Client Setup
s3 = boto3.client("s3")
BUCKET = "mlops-prediction-56"

# Exact S3 Key Path
KEY = "processed/2026-10-05/Mlops_house_predication_clean_v1.csv"  # Update with the actual date and file name

def fetch_data():
    obj = s3.get_object(Bucket=BUCKET, Key=KEY)
    df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
    return df

# 2. Fetch and Prepare Data
df = fetch_data()
print(f"Fetched shape: {df.shape}")

# Features & Target Columns
X = df[["sqft", "bedrooms", "bathrooms", "age_years", "garage", "location_score"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. MLflow Tracking & Experiment Setup
mlflow.set_tracking_uri("http://100.60.192.109:5000")
mlflow.set_experiment("mlops-house-prediction")

with mlflow.start_run():
    n_estimators = 150
    max_depth = 8

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predictions & Evaluation
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # Logging Parameters & Metrics
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("data_source", f"s3://{BUCKET}/{KEY}")
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    # Log & Register Model to MLflow Registry (Fixing skops trust verification)
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name="house-price-predictor",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    print(f"\nMAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")
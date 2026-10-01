from io import StringIO
import os
import boto3
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

s3 = boto3.client("s3")

BUCKET = "mlops-prediction-56"
KEY = "Mlops_house_predication_raw_data.csv"


def fetch_data():
    response = s3.get_object(Bucket=BUCKET, Key=KEY)
    df = pd.read_csv(StringIO(response["Body"].read().decode("utf-8")))
    return df


df = fetch_data()

# Features / Target separation
X = df[["sqft", "bedrooms", "bathrooms", "age_years", "garage", "location_score"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fix artifact path issue by explicitly setting local mlruns folder
mlflow.set_tracking_uri("sqlite:///mlflow.db")
os.makedirs("./mlruns", exist_ok=True)

exp_name = "mlops-house-prediction"
experiment = mlflow.get_experiment_by_name(exp_name)
if experiment is None:
    mlflow.create_experiment(exp_name, artifact_location="./mlruns")
mlflow.set_experiment(exp_name)

with mlflow.start_run():
    n_estimators = 150
    max_depth = 8

    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # Log metrics & params
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    # Log model artifact with trusted types
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    print(f"\nMAE: {mae:.4f} | RMSE: {rmse:.4f} | R2 Score: {r2:.4f}")
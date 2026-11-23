import hashlib
from pathlib import Path
from mlflow.tracking import MlflowClient
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml

from src.mp253.clean import FEATURE_COLS, TARGET_COL, clean_dataframe

NUMERIC_COLS = [
    "distance_km",
    "tiffins",
    "handovers",
    "wait_at_station_min",
    "rider_age",
    "temp_c",
]
CATEGORICAL_COLS = ["transport", "route_zone"]


def compute_sha256(file_path: Path) -> str:
  return hashlib.sha256(file_path.read_bytes()).hexdigest()


def build_preprocessor() -> ColumnTransformer:
  num_pipe = Pipeline([
      ("imputer", SimpleImputer(strategy="median")),
      ("scaler", StandardScaler()),
  ])
  cat_pipe = Pipeline([
      ("imputer", SimpleImputer(strategy="most_frequent")),
      ("ohe", OneHotEncoder(handle_unknown="ignore")),
  ])
  return ColumnTransformer([
      ("num", num_pipe, NUMERIC_COLS),
      ("cat", cat_pipe, CATEGORICAL_COLS),
  ])


def train_v1(params_path: str = "params.yaml") -> dict:
  with open(params_path, "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

  raw_train_path = Path(params["data"]["train_path"])
  later_path = Path(params["data"]["later_path"])
  data_sha256 = compute_sha256(raw_train_path)

  raw_train = pd.read_csv(raw_train_path)
  clean_train = clean_dataframe(raw_train, params, is_train=True)
  later_df = clean_dataframe(pd.read_csv(later_path), params, is_train=False)

  X = clean_train[FEATURE_COLS]
  y = clean_train[TARGET_COL]

  X_train, X_val, y_train, y_val = train_test_split(
      X,
      y,
      test_size=params["model"].get("test_size", 0.2),
      random_state=params["model"].get("random_state", 42),
  )

  mlflow.set_tracking_uri("sqlite:///mlflow.db")
  experiment_name = params["model"].get("experiment_name", "mp253_tiffin_relay")
  registered_name = params["model"].get("name", "mp253-tiffin-delivery")
  mlflow.set_experiment(experiment_name)

  candidates = [
      (
          "ridge_alpha_1.0",
          Ridge(alpha=1.0, random_state=params["model"].get("random_state", 42)),
          {"model_type": "Ridge", "alpha": 1.0},
      ),
      (
          "rf_n100_d8",
          RandomForestRegressor(
              n_estimators=params["model"].get("n_estimators", 100),
              max_depth=params["model"].get("max_depth", 8),
              random_state=params["model"].get("random_state", 42),
              n_jobs=-1,
          ),
          {
              "model_type": "RandomForestRegressor",
              "n_estimators": params["model"].get("n_estimators", 100),
              "max_depth": params["model"].get("max_depth", 8),
          },
      ),
      (
          "gbr_n150_d4",
          GradientBoostingRegressor(
              n_estimators=150,
              max_depth=4,
              learning_rate=0.05,
              random_state=params["model"].get("random_state", 42),
          ),
          {
              "model_type": "GradientBoostingRegressor",
              "n_estimators": 150,
              "max_depth": 4,
              "learning_rate": 0.05,
          },
      ),
  ]

  best_mae = float("inf")
  best_run_id = None
  best_pipeline = None

  for run_name, estimator, extra_params in candidates:
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", estimator),
    ])
    pipe.fit(X_train, y_train)

    val_preds = pipe.predict(X_val)
    later_preds = pipe.predict(later_df[FEATURE_COLS])

    mae_val = float(mean_absolute_error(y_val, val_preds))
    mae_later = float(mean_absolute_error(later_df[TARGET_COL], later_preds))

    with mlflow.start_run(run_name=run_name) as run:
      mlflow.log_param("data_sha256", data_sha256)
      for k, v in extra_params.items():
        mlflow.log_param(k, v)
      mlflow.log_metric("mae_val", mae_val)
      mlflow.log_metric("mae_later", mae_later)
      try:
        mlflow.sklearn.log_model(pipe, artifact_path="model")
      except Exception:
        mlflow.sklearn.log_model(
            pipe, artifact_path="model", serialization_format="cloudpickle"
        )

      print(
          f"Run {run_name} ({run.info.run_id}): mae_val={mae_val:.4f},"
          f" mae_later={mae_later:.4f}"
      )

      if mae_later < best_mae:
        best_mae = mae_later
        best_run_id = run.info.run_id
        best_pipeline = pipe

  # Register the best run as version 1 and set @champion
  model_uri = f"runs:/{best_run_id}/model"
  mv = mlflow.register_model(model_uri=model_uri, name=registered_name)
  client = MlflowClient()
  client.set_registered_model_alias(registered_name, "champion", mv.version)

  models_dir = Path("models")
  models_dir.mkdir(parents=True, exist_ok=True)
  import joblib

  joblib.dump(best_pipeline, models_dir / "model.joblib")

  print(
      f"\nChampion registered: name={registered_name}, version={mv.version},"
      f" run_id={best_run_id}, mae_later={best_mae:.4f}"
  )
  return {
      "experiment_name": experiment_name,
      "best_run_id": best_run_id,
      "registered_model_name": registered_name,
      "champion_version": str(mv.version),
  }


if __name__ == "__main__":
  train_v1()
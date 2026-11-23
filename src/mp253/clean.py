from pathlib import Path
import numpy as np
import pandas as pd
import yaml

FEATURE_COLS = [
    "distance_km",
    "tiffins",
    "handovers",
    "wait_at_station_min",
    "rider_age",
    "temp_c",
    "transport",
    "route_zone",
]
TARGET_COL = "delivery_min"


def load_params(params_path: str = "params.yaml") -> dict:
  with open(params_path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f)


def clean_dataframe(df: pd.DataFrame, params: dict, is_train: bool = True) -> pd.DataFrame:
  out = df.copy()
  data_cfg = params["data"]

  # 1. Fix unit mixup on wait_at_station_min for affected route_zone
  factor = data_cfg.get("unit_mixup_factor", 60)
  zone = data_cfg.get("unit_mixup_zone", "Old City")
  if {"wait_at_station_min", "route_zone"}.issubset(out.columns):
    mask_zone = out["route_zone"] == zone
    out.loc[mask_zone, "wait_at_station_min"] = (
        pd.to_numeric(out.loc[mask_zone, "wait_at_station_min"], errors="coerce")
        / factor
    )

  # 2. Replace sentinel value (-1) in handovers with NaN
  sentinel = data_cfg.get("sentinel_handovers", -1)
  if "handovers" in out.columns:
    out["handovers"] = pd.to_numeric(out["handovers"], errors="coerce")
    out.loc[out["handovers"] == sentinel, "handovers"] = np.nan

  # 3. Fix target typos (values typed ~10x too large) on training data
  if is_train and TARGET_COL in out.columns:
    threshold = data_cfg.get("target_typo_threshold", 150.0)
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce")
    typo_mask = out[TARGET_COL] > threshold
    out.loc[typo_mask, TARGET_COL] = out.loc[typo_mask, TARGET_COL] / 10.0

  return out


def run_cleaning(params_path: str = "params.yaml") -> pd.DataFrame:
  params = load_params(params_path)
  raw_path = Path(params["data"]["train_path"])
  out_path = Path(params["data"]["cleaned_train_path"])
  out_path.parent.mkdir(parents=True, exist_ok=True)

  raw_df = pd.read_csv(raw_path)
  cleaned_df = clean_dataframe(raw_df, params, is_train=True)
  cleaned_df.to_csv(out_path, index=False)
  return cleaned_df
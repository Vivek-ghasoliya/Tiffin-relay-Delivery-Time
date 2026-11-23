import numpy as np
import pandas as pd
from src.mp253.clean import FEATURE_COLS, TARGET_COL, clean_dataframe


def sample_raw_df() -> pd.DataFrame:
  return pd.DataFrame({
      "record_id": ["R1", "R2", "R3", "R4"],
      "date": ["2026-01-10", "2026-02-15", "2026-03-20", "2026-04-05"],
      "distance_km": [5.0, 12.0, 8.0, 15.0],
      "tiffins": [20, 35, 15, 40],
      "handovers": [2, -1, 3, 1],
      "wait_at_station_min": [10.0, 600.0, 8.0, 12.0],
      "rider_age": [25, 40, 32, 29],
      "temp_c": [28.0, 34.0, 26.0, 31.0],
      "transport": ["scooter", "local_train", "bicycle", "scooter"],
      "route_zone": ["Central", "Old City", "Suburb", "Central"],
      "delivery_min": [25.0, 45.0, 30.0, 420.0],
  })


def test_clean_fixes_unit_mixup_sentinel_and_target_typos():
  params = {
      "data": {
          "unit_mixup_factor": 60,
          "unit_mixup_zone": "Old City",
          "sentinel_handovers": -1,
          "target_typo_threshold": 150.0,
      }
  }
  cleaned = clean_dataframe(sample_raw_df(), params, is_train=True)

  # Old City wait_at_station_min divided by 60 (600 -> 10.0)
  old_city_wait = cleaned.loc[
      cleaned["route_zone"] == "Old City", "wait_at_station_min"
  ].iloc[0]
  assert np.isclose(old_city_wait, 10.0)

  # Handovers == -1 replaced with NaN
  assert pd.isna(cleaned.loc[1, "handovers"])

  # Target typo > 150 divided by 10 (420.0 -> 42.0)
  assert np.isclose(cleaned.loc[3, TARGET_COL], 42.0)


def test_clean_preserves_feature_columns():
  params = {"data": {}}
  cleaned = clean_dataframe(sample_raw_df(), params, is_train=False)
  for col in FEATURE_COLS:
    assert col in cleaned.columns


def test_clean_does_not_mutate_input():
  params = {"data": {"target_typo_threshold": 150.0}}
  raw = sample_raw_df()
  raw_copy = raw.copy(deep=True)
  _ = clean_dataframe(raw, params, is_train=True)
  pd.testing.assert_frame_equal(raw, raw_copy)
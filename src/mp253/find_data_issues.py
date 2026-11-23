from pathlib import Path
import pandas as pd

train = pd.read_csv(Path("data/raw/train.csv"))
later = pd.read_csv(Path("data/raw/later.csv"))
holdout = pd.read_csv(Path("data/raw/holdout_features.csv"))

print("Shape (train, later, holdout):", train.shape, later.shape, holdout.shape)
print("Columns in train:", list(train.columns))
print("Columns in holdout:", list(holdout.columns))

print("\n1. LEAKING COLUMN (in train but not holdout):")
print("  ", list(set(train.columns) - set(holdout.columns)))

print("\n2. DUPLICATE ROWS (extra duplicate rows in train):")
print("  ", int(train.duplicated().sum()))

print("\n3. TARGET TYPOS (top 15 delivery_min values):")
if "delivery_min" in train.columns:
  print(
      "  ", sorted(train["delivery_min"].dropna().tolist(), reverse=True)[:15]
  )

print("\n4. SENTINEL VALUES (negative or >=999 values in numeric columns):")
num_cols = [
    c for c in train.select_dtypes(include="number").columns if c != "delivery_min"
]
for col in num_cols:
  s = train[col].dropna()
  neg = s[s < 0].value_counts().head(5).to_dict()
  high = s[s >= 999].value_counts().head(5).to_dict()
  print(
      f"  {col}: min={s.min()}, max={s.max()}, <0={neg}, >=999={high}"
  )

print("\n5. UNIT MIXUP (min / mean / max per route_zone):")
if "route_zone" in train.columns:
  check_cols = num_cols + (
      ["delivery_min"] if "delivery_min" in train.columns else []
  )
  for col in check_cols:
    grp = train.groupby("route_zone", dropna=False)[col].agg(
        ["min", "mean", "max"]
    )
    print(f"\n  [{col} by route_zone]\n{grp}")

print("\n6. SPELLING VARIANTS (train vs later value counts):")
for col in ["transport", "route_zone"]:
  if col in train.columns:
    print(f"  {col} (train): {train[col].value_counts(dropna=False).to_dict()}")
    if col in later.columns:
      print(f"  {col} (later): {later[col].value_counts(dropna=False).to_dict()}")

print("\n7. STUCK SENSOR (longest identical runs sorted by date):")
if "date" in train.columns:
  df_sorted = train.sort_values("date").reset_index(drop=True)
  for col in num_cols:
    runs = df_sorted[col].ne(df_sorted[col].shift()).cumsum()
    top_runs = (
        df_sorted.groupby(runs)
        .agg(
            val=(col, "first"),
            count=(col, "size"),
            start_date=("date", "first"),
            end_date=("date", "last"),
        )
        .sort_values("count", ascending=False)
        .head(3)
    )
    print(f"  {col}:\n{top_runs.to_string(index=False)}")
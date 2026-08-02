import pandas as pd

df = pd.read_parquet('data/processed/protein_features_hol.parquet')
serving = df[df["date"] >= "2017-07-31"].copy()
serving.to_parquet('data/serving/features.parquet', index=False)
print(f"Serving data: {len(serving):,} rows")
print(f"Date range: {serving['date'].min().date()} to {serving['date'].max().date()}")

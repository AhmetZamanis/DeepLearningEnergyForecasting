import pandas as pd
import numpy as np

from pathlib import Path
#from src.utils import get_root_dir


print("Starting training data update script...")

print("Getting directories...")
#work_dir = get_root_dir()
work_dir = Path.cwd()
data_dir = work_dir / "data" / "deployment"

raw_filename = data_dir / "raw" / "consumption.csv"
if (raw_filename.exists()) == False:
        raise Exception("Raw data not found in data/deployment/raw. Run raw data update script.")

processed_filename = data_dir / "processed" / "training_data.csv"

print("Getting raw consumption data...")
df = pd.read_csv(raw_filename).drop("time", axis = 1)
df["date"] = pd.to_datetime(df["date"], format = "ISO8601")

print("Adding time features...")
df["trend"] = df.index.values

hourofday = df.date.dt.hour
df["hour_sin"] = np.sin(2 * np.pi * hourofday / 24)
df["hour_cos"] = np.cos(2 * np.pi * hourofday / 24)

dayofweek = df.date.dt.dayofweek
df["day_sin"] = np.sin(2 * np.pi * dayofweek / 7)
df["day_cos"] = np.cos(2 * np.pi * dayofweek / 7)

month = df.date.dt.month
df["month_sin"] = np.sin(2 * np.pi * month / 12)
df["month_cos"] = np.cos(2 * np.pi * month / 12)

print("Writing training data to: /data/deployment/processed")
df.to_csv(processed_filename, index = False)

print("Training data updated.")
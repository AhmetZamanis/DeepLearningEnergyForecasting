import hydra
import pandas as pd
import os

from dotenv import load_dotenv
from pathlib import Path
from omegaconf import DictConfig
from math import ceil
from src.data_extraction.api import get_tgt, get_consumption_data, _get_request_dates
#from src.utils import get_root_dir


print("Starting raw consumption data update script...")

@hydra.main(version_base = None, config_path = "configs", config_name = "config")
def update_raw_data(cfg: DictConfig) -> None:

    print("Getting consumption data request configs...")
    years_of_data = cfg.data.years_of_data
    timeout = cfg.data.timeout

    print(f"Years of data: {years_of_data}")
    print(f"Timeout between data requests: {timeout} seconds")

    print("Getting directories...")
    #work_dir = get_root_dir()
    work_dir = Path.cwd()
    data_dir = work_dir / "data" / "deployment" / "raw"
    raw_filename = data_dir / "consumption.csv"

    print("Checking for existing raw data...")
    old_data_exists = False

    # Try to load "consumption.csv". Pass if it doesn't exist.
    try:
        df_old = pd.read_csv(raw_filename)
        df_old.loc[:, "date"] = pd.to_datetime(df_old["date"], format = "ISO8601")
        old_data_exists = True
        
    except:
        print("No existing raw data.")
        pass

    if old_data_exists:

        # Check if requested years of data is long enough to prevent gaps
        # "years_of_data" must be equal or longer to the difference between the ends of new & old data
        new_start_dates, new_end_date = _get_request_dates(years_of_data)
        new_end_date = pd.to_datetime(new_end_date, format = "ISO8601")
        old_end_date = df_old["date"].iloc[-1]
        diff_seconds = pd.Timedelta((new_end_date - old_end_date)).total_seconds()
        diff_years = ceil(diff_seconds / (365.2425 * 24 * 60 * 60))  # Seconds to years conversion, factoring in leap years

        if years_of_data < diff_years:
            raise Exception(f"The request will result in gaps in the data. Ensure 'years_of_data' >= {diff_years}.")

        # Check if data start date will change after the update
        new_start_date = pd.to_datetime(new_start_dates[-1], format = "ISO8601")
        old_start_date = df_old["date"].iloc[0]
        
        if new_start_date < old_start_date:
            print("WARNING: The data start date will change after the update. Make sure to update the training data & to retrain the model.")
        
    print("Getting credentials from .env ...")
    load_dotenv()
    epias_username = os.getenv("epias_username")
    epias_password = os.getenv("epias_password")

    print("Requesting TGT...")
    tgt = get_tgt(epias_username, epias_password)

    print("Requesting consumption raw data...")
    df_new = get_consumption_data(tgt, years_of_data, timeout)

    if old_data_exists:
        print("Merging with existing data...")
        df = pd.concat([df_old, df_new]).drop_duplicates(subset = ["date"], keep = "last", ignore_index = True)
        
    else:
        df = df_new.copy()

    print("Writing raw consumption data to: /data/deployment/raw")
    df = df.sort_values("date")
    df.to_csv(raw_filename, index = False)

if __name__ == "__main__":
    update_raw_data()

print("Raw consumption data updated.")
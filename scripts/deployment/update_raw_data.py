import hydra
import pandas as pd
import os

from dotenv import load_dotenv
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from src.data_extraction.api import get_tgt, get_consumption_data
from src.utils import get_root_dir


print("Starting raw consumption data update script...")

@hydra.main(version_base = None, config_path = "configs", config_name = "config")
def update_raw_data(cfg: DictConfig) -> None:

    print("Getting consumption data configs...")
    years_of_data = cfg.data.years_of_data
    timeout = cfg.data.timeout

    print(f"Years of data: {years_of_data}")
    print(f"Timeout between data requests: {timeout} seconds")

    print("Getting directories...")
    root_dir = get_root_dir()
    data_dir = root_dir / "data" / "deployment" / "raw"
    filepath = data_dir / "consumption.csv"

    print("Getting credentials from .env ...")
    load_dotenv()
    epias_username = os.getenv("epias_username")
    epias_password = os.getenv("epias_password")

    print("Requesting TGT...")
    tgt = get_tgt(epias_username, epias_password)

    print("Requesting consumption data...")
    df = get_consumption_data(tgt, years_of_data, timeout)

    print("Writing raw consumption data to: /data/deployment/raw")
    df.to_csv(filepath, index = False)

    print("Raw consumption data updated.")

if __name__ == "__main__":
    update_raw_data()
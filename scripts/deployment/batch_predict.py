import datetime
import hydra
import numpy as np
import pandas as pd
import torch
import lightning as L
import glob
import warnings

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pathlib import Path
from joblib import load
from src.deep_learning.preprocessing import SequenceDataset
from src.deep_learning.prediction import predictions_to_dataframe
from src.deep_learning.transformer import LITransformer
#from src.utils import get_root_dir


print("Starting transformer batch prediction script...")

@hydra.main(version_base = None, config_path = "configs", config_name = "config")
def batch_predict(cfg: DictConfig) -> None:

    print("Getting directories...")
    #work_dir = get_root_dir()
    work_dir = Path.cwd()
    data_dir = work_dir / "data" / "deployment"
    model_dir = work_dir / "models" / "deployment"
    transformer_dir = model_dir / "transformer"
    scaler_dir = model_dir / "scaler"
    preds_dir = data_dir / "predictions"

    processed_filename = data_dir / "processed" / "training_data.csv"
    if (processed_filename.exists()) == False:
        raise Exception("Training data not found in data/deployment/processed. Run training data update script.")

    print("Getting transformer batch prediction configs...")
    source_length = cfg.transformer.source_length
    target_length = cfg.transformer.target_length
    forecast_t = cfg.transformer.forecast_t
    num_workers = cfg.transformer.num_workers
    accelerator = cfg.torch.accelerator
    precision = cfg.torch.precision
    matmul_precision = cfg.torch.matmul_precision
    model_summary = cfg.training.model_summary
    progress_bar = cfg.training.progress_bar

    print("Overridden configs:")
    for key, value in HydraConfig.get().items():
        if key.startswith("overrides"):
            print(f"{key}: {value}")

    print("Getting last trained transformer model...")
    transformer_pattern = (transformer_dir / "*.ckpt").__str__()
    transformer_paths = glob.glob(transformer_pattern)

    if len(transformer_paths) == 0:
        raise Exception("No trained transformer model found in models/deployment/transformer. Run model training script.")

    latest_transformer = sorted(transformer_paths)[-1] 
    model = LITransformer.load_from_checkpoint(latest_transformer)

    print("Getting last fitted scaler...")
    scaler_pattern = (scaler_dir / "*.joblib").__str__()
    scaler_paths = glob.glob(scaler_pattern)

    if len(scaler_paths) == 0:
        raise Exception("No fitted scaler found in models/deployment/scaler. Run model training script.")

    latest_scaler = sorted(scaler_paths)[-1] 
    scaler = load(latest_scaler)

    print("Getting prediction data...")
    df_source = pd.read_csv(processed_filename)
    df_source["date"] = pd.to_datetime(df_source["date"], format = "ISO8601")

    # Get last L steps of training data, source sequence
    df_source = df_source.iloc[-source_length:, :]

    # Extend into next H steps, target sequence
    last_date = df_source["date"].max()
    new_dates = pd.date_range(
        last_date, 
        periods = target_length + 1, 
        freq = "h",
        inclusive = "right"
    )
    df_target = pd.DataFrame(new_dates, columns = ["date"])

    # Add mock "consumption" column for compatibility with SequenceScaler
    df_target["consumption"] = 0

    # Add time features to target sequence
    first_trend = df_source["trend"].iloc[-1] + 1
    df_target["trend"] = pd.Series(
        range((first_trend), (first_trend + target_length), 1)
    )

    hourofday = df_target.date.dt.hour
    df_target["hour_sin"] = np.sin(2 * np.pi * hourofday / 24)
    df_target["hour_cos"] = np.cos(2 * np.pi * hourofday / 24)

    dayofweek = df_target.date.dt.dayofweek
    df_target["day_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    df_target["day_cos"] = np.cos(2 * np.pi * dayofweek / 7)

    month = df_target.date.dt.month
    df_target["month_sin"] = np.sin(2 * np.pi * month / 12)
    df_target["month_cos"] = np.cos(2 * np.pi * month / 12)

    print("Performing feature scaling...")

    # Set datetime index to drop date from array data, but still keep it for later
    df_source = df_source.set_index("date")
    df_target = df_target.set_index("date")
    
    # Get data as arrays, put them in lists for SequenceScaler
    # Get rid of mock "consumption" column in target sequence after scaling
    pred_data = SequenceDataset(
        scaler.transform([df_source.values]),
        scaler.transform([df_target.values])[:, :, 1:], 
    )

    # Create Torch dataloader
    shuffle = False
    pred_loader = torch.utils.data.DataLoader(
        pred_data, batch_size = 1, num_workers = num_workers, shuffle = shuffle
    )

    # Set Torch settings
    torch.set_float32_matmul_precision(matmul_precision)
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*no `val_dataloader`*")

    # Create trainer
    trainer = L.Trainer(
        accelerator = accelerator,  
        devices = "auto",
        precision = precision,
        enable_model_summary = model_summary,
        logger = False,
        enable_progress_bar = progress_bar,
        enable_checkpointing = False
    )

    print("Making predictions...")
    preds_raw = trainer.predict(model, pred_loader)
    preds = torch.cat(preds_raw, dim = 0).cpu().numpy().astype(np.float32)
    preds = scaler.backtransform_preds(preds)

    # Combine back with dates
    df_preds = predictions_to_dataframe(preds, [df_target])

    print("Writing predictions to: /data/deployment/predictions")
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    preds_path = preds_dir / f"{current_date}.csv"
    df_preds.to_csv(preds_path, index = False)

if __name__ == "__main__":
    batch_predict()

print("Transformer batch prediction complete.")
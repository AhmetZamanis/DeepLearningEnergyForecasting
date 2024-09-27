import datetime
import hydra
import pandas as pd
import torch
import lightning as L
import glob
import warnings

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pathlib import Path
from joblib import dump
from src.deep_learning.preprocessing import get_transformer_sequences, SequenceScaler, SequenceDataset
from src.deep_learning.transformer import LITransformer
#from src.utils import get_root_dir


print("Starting transformer model training script...")

@hydra.main(version_base = None, config_path = "configs", config_name = "config")
def train_model(cfg: DictConfig) -> None:

    print("Getting directories...")
    #work_dir = get_root_dir()
    work_dir = Path.cwd()
    data_dir = work_dir / "data" / "deployment"
    model_dir = work_dir / "models" / "deployment"

    processed_filename = data_dir / "processed" / "training_data.csv"
    if (processed_filename.exists()) == False:
        raise Exception("Training data not found in data/deployment/processed. Run training data update script.")

    tuning_dir = data_dir / "tuning-logs"
    tuning_log_pattern = (tuning_dir / "transformer_*.csv").__str__()  # Convert from Path to str for glob

    print("Getting transformer model best tune...")
    # glob takes the str-converted filepath pattern, returns filepaths in the same format
    # The returned str filepath is used to load the tune
    # Should work on all systems, because the str conversion is made by Path, locally
    tuning_logs = glob.glob(tuning_log_pattern) 

    if (len(tuning_logs) == 0):
        raise Exception("No tuning log found in data/deployment/tuning-logs. Run model tuning script.")

    latest_log = sorted(tuning_logs)[-1] 
    best_tune = pd.read_csv(latest_log).query("state == 'COMPLETE'").iloc[0, :]

    print("Getting training data...")
    df = pd.read_csv(processed_filename)
    df["date"] = pd.to_datetime(df["date"], format = "ISO8601")
    
    print("Getting transformer model training configs...")
    source_length = cfg.transformer.source_length
    target_length = cfg.transformer.target_length
    forecast_t = cfg.transformer.forecast_t
    horizon_start = cfg.transformer.horizon_start
    quantiles = cfg.transformer.quantiles
    batch_size = cfg.transformer.batch_size
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

    print("Sequencing training data...")
    
    # Drop last L steps, as potential prediction source sequence
    df = df.iloc[0:-source_length, :]
    
    source_sequences, target_sequences = get_transformer_sequences(
        df, source_length, target_length, forecast_t
    )

    print("Performing feature scaling...")
    scaler = SequenceScaler()
    _ = scaler.fit(source_sequences, target_sequences)

    train_data = SequenceDataset(
        scaler.transform(source_sequences),
        scaler.transform(target_sequences),
    )

    # Create Torch dataloader
    shuffle = False
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size = batch_size, num_workers = num_workers, shuffle = shuffle
    )

    # Create hyperparameters dict
    hyperparameters_dict = {
        "source_length": source_length,
        "target_length": target_length,
        "horizon_start": horizon_start,
        "quantiles": quantiles,
        "d_model": 2 ** best_tune["params_d_model"],
        "n_heads": 2 ** best_tune["params_n_heads"],
        "n_encoders": best_tune["params_n_encoders"],
        "n_decoders": best_tune["params_n_encoders"],
        "d_feedforward": 2 ** best_tune["params_d_feedforward"],
        "activation": best_tune["params_activation"],
        "learning_rate": best_tune["params_learning_rate"],
        "lr_decay": best_tune["params_lr_decay"],
        "dropout_rate": best_tune["params_dropout_rate"]
    }

    # Set Torch settings
    torch.set_float32_matmul_precision(matmul_precision)
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*no `val_dataloader`*")

    # Create trainer
    trainer = L.Trainer(
        accelerator = accelerator,  
        devices = "auto",
        precision = precision,
        max_epochs = int(best_tune["user_attrs_n_epochs"]),
        enable_model_summary = model_summary,
        logger = False,
        enable_progress_bar = progress_bar,
        enable_checkpointing = False
    )
    
    print("Training transformer model...")
    model = LITransformer(hyperparameters_dict)
    trainer.fit(model, train_loader)

    print("Saving trained transformer model to: /models/deployment/transformer")
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path = model_dir / "transformer" / f"{current_date}.ckpt"
    trainer.save_checkpoint(model_path)

    print("Saving fitted scaler to: /models/deployment/scaler")
    scaler_path = model_dir / "scaler" / f"{current_date}.joblib"
    dump(scaler, scaler_path)

if __name__ == "__main__":
    train_model()

print("Transformer model training complete.")
import datetime
import hydra
import pandas as pd
import numpy as np
import torch
import lightning as L
import optuna
import warnings

from lightning.pytorch.callbacks import Callback
from optuna.integration import PyTorchLightningPruningCallback
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from src.deep_learning.preprocessing import get_transformer_sequences, SequenceScaler, SequenceDataset
from src.deep_learning.testing import train_val_split
from src.deep_learning.transformer import LITransformer
from src.utils import get_root_dir


print("Starting transformer model tuning script...")

@hydra.main(version_base = None, config_path = "configs", config_name = "config")
def tune_model(cfg: DictConfig) -> None:

    print("Getting transformer model & tuning configs...")
    source_length = cfg.transformer.source_length
    target_length = cfg.transformer.target_length
    forecast_t = cfg.transformer.forecast_t
    horizon_start = cfg.transformer.horizon_start
    quantiles = cfg.transformer.quantiles
    batch_size = cfg.transformer.batch_size
    num_workers = cfg.transformer.num_workers
    val_size = cfg.tuning.val_size
    tolerance = cfg.tuning.tolerance
    patience = cfg.tuning.patience
    max_epochs = cfg.tuning.max_epochs
    n_trials = cfg.tuning.n_trials

    # Set Torch settings
    #torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('medium')  # MAKE ADJUSTABLE??
    #L.seed_everything(random_seed, workers = True)
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    print("Overridden configs:")
    for key, value in HydraConfig.get().items():
        if key.startswith("overrides"):
            print(f"{key}: {value}")

    print("Getting directories...")
    work_dir = get_root_dir()
    #work_dir = Path.cwd()
    data_dir = work_dir / "data" / "deployment"
    processed_dir = data_dir / "processed" / "training_data.csv"
    tuning_dir = data_dir / "tuning-logs"

    print("Getting & sequencing training data...")
    df = pd.read_csv(processed_dir)
    df["date"] = pd.to_datetime(df["date"], format = "ISO8601")

    # Drop last L steps, as potential prediction source sequence
    df = df.iloc[0:-source_length, :]

    # Sequence training data
    source_sequences, target_sequences = get_transformer_sequences(
        df, source_length, target_length, forecast_t
    )

    print("Performing train-validation split...")
    train_source, train_target, val_source, val_target = train_val_split(
        source_sequences, target_sequences, (1-val_size), batch_size
    )

    # # TEST: Write last sequences, check if they're good
    # print(len(train_source))
    # train_source[-1].to_csv(tuning_dir / "train_source.csv")
    # train_target[-1].to_csv(tuning_dir / "train_target.csv")
    # val_source[-1].to_csv(tuning_dir / "val_source.csv")
    # val_target[-1].to_csv(tuning_dir / "val_target.csv")

    print("Performing feature scaling...")
    scaler = SequenceScaler()
    _ = scaler.fit(train_source, train_target)

    train_data = SequenceDataset(
        scaler.transform(train_source),
        scaler.transform(train_target),
    )

    val_data = SequenceDataset(
        scaler.transform(val_source),
        scaler.transform(val_target),
    )

    print("Creating Torch dataloaders...")
    shuffle = False
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size = batch_size, num_workers = num_workers, shuffle = shuffle
    )

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size = batch_size, num_workers = num_workers, shuffle = shuffle
    )

    print("Performing Optuna study...")

    # Define Optuna objective
    def objective_transformer(trial):
    
        # Define search space
        n_heads = 2 ** trial.suggest_int("n_heads", 1, 3)  # Powers of 2, 2 to 8
        
        # In the PyTorch transformer, embed_dim has to be divisible by n_heads.
        d_model = max(
            n_heads, 
            2 ** trial.suggest_int("d_model", 2, 6)  # Powers of 2, 4 to 64
        ) 
    
        d_feedforward = 2 ** trial.suggest_int("d_feedforward", 2, 6)  # Powers of 2, 4 to 64
        n_encoders = trial.suggest_int("n_encoders", 1, 3)  # 1 to 3
        activation = trial.suggest_categorical("activation", ["relu", "gelu"])
        learning_rate = trial.suggest_float("learning_rate", 5e-4, 5e-2)  # 0.0005 to 0.05
        lr_decay = trial.suggest_float("lr_decay", 0.9, 1)
        dropout_rate = trial.suggest_float("dropout_rate", 0, 0.1)
    
        # Probably best to match n. of encoder & decoder layers
        n_decoders = n_encoders
    
        # Create hyperparameters dict
        hyperparameters_dict = {
            "source_length": source_length,
            "target_length": target_length,
            "horizon_start": horizon_start,
            "quantiles": quantiles,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_encoders": n_encoders,
            "n_decoders": n_decoders,
            "d_feedforward": d_feedforward,
            "activation": activation,
            "learning_rate": learning_rate,
            "lr_decay": lr_decay,
            "dropout_rate": dropout_rate
        }
    
        # Create early stop callback
        callback_earlystop = L.pytorch.callbacks.EarlyStopping(
            monitor = "val_loss", 
            mode = "min", 
            min_delta = tolerance, 
            patience = patience
        )
    
        # Create pruning callback
        callback_pruner = PyTorchLightningPruningCallback(trial, monitor = "val_loss")
    
        # Create trainer
        trainer = L.Trainer(
            max_epochs = max_epochs,

            # MAKE THESE ADJUSTABLE?
            accelerator = "gpu",  
            devices = "auto",
            precision = "16-mixed",
            
            callbacks = [callback_earlystop, callback_pruner],
            enable_model_summary = False,
            logger = False,
            enable_progress_bar = False,
            enable_checkpointing = False
        )
    
        # Create & train model
        model = LITransformer(hyperparameters_dict)
        trainer.fit(model, train_loader, val_loader)
    
        # Retrieve best val score and n. of epochs
        score = callback_earlystop.best_score.cpu().numpy()
        epoch = trainer.current_epoch - callback_earlystop.wait_count  # Starts from 1
    
        # Report best n. of epochs to study
        trial.set_user_attr("n_epochs", epoch)
      
        return score

    # Create study
    study_transformer = optuna.create_study(
        sampler = optuna.samplers.TPESampler(),
        pruner = optuna.pruners.HyperbandPruner(),
        study_name = "tune_transformer",
        direction = "minimize"
    )

    # Optimize study
    study_transformer.optimize(objective_transformer, n_trials = n_trials, show_progress_bar = True)

    print("Saving tuning logs to: /data/deployment/tuning-logs")

    # Export study
    current_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logname = f"transformer-{current_date}.csv"
    study_dir = tuning_dir / logname
    trials_transformer = study_transformer.trials_dataframe().sort_values("value", ascending = True)
    trials_transformer.to_csv(study_dir, index = False)

if __name__ == "__main__":
    tune_model()

print("Transformer model tuning complete.")
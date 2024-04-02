# Torch classes & functions used in notebook 3.2, for the Transformer model.
import pandas as pd
import numpy as np
import torch
import lightning as L


class sequence_scaler:
    """
    Takes in lists of dataframes where each dataframe is a source or target sequence.

    Returns scaled 3D numpy arrays of shape (observations, timesteps, features).
    Can also backtransform scaled predictions.
    """

    def __init__(self, feature_range = (-1, 1)):
        self.lower = feature_range[0]
        self.upper = feature_range[1]

    def fit(self, source_df, target_df):

        # Get source & target sequences as 3D arrays
        # The time index will be skipped, yielding shape (N, seq_length, seq_dims)
        source = np.stack(source_df, axis = 0)
        target = np.stack(target_df, axis = 0)

        # Get number of dimensions
        self.num_dimensions = source.shape[2]
        
        # Extract & save minimum, maximum for each dimension
        dimensions_mini = []
        dimensions_maxi = []
        for dimension in range(0, self.num_dimensions):
            min = np.min([
                np.min(source[:, :, dimension]),
                np.min(target[:, :, dimension])
            ])
            dimensions_mini.append(min)

            max = np.max([
                np.max(source[:, :, dimension]),
                np.max(target[:, :, dimension])
            ])
            dimensions_maxi.append(max)

        self.dimensions_mini = dimensions_mini
        self.dimensions_maxi = dimensions_maxi

    def transform(self, scale_df):

        # Get sequence as 3D arrays
        scale_array = np.stack(scale_df, axis = 0)

        # Initialize list of scaled dimensions
        scaled_dimensions = []

        # Scale each dimension & append to list
        for dimension in range(0, self.num_dimensions):
            values = scale_array[:, :, dimension]
            min = self.dimensions_mini[dimension]
            max = self.dimensions_maxi[dimension]
            std = (values - min) / (max - min)
            scaled = std * (self.upper - self.lower) + self.lower
            scaled_dimensions.append(scaled)

        # Stack over 3rd axis & return
        return np.stack(scaled_dimensions, axis = 2)

    def backtransform_preds(self, preds_array, fitted_preds_dim = 0):

        # Get n. of predicted quantiles to backtransform
        n_quantiles = preds_array.shape[-1]

        # Get the fitted mini & maxi for predictions
        min = self.dimensions_mini[fitted_preds_dim] 
        max = self.dimensions_maxi[fitted_preds_dim]

        # Initialize list of backtransformed quantiles
        backtrafo_quantiles = []

        # Backtransform each quantile & append to list
        for quantile in range(0, n_quantiles):
            scaled = preds_array[:, :, quantile]
            std = (scaled - self.lower) / (self.upper - self.lower)
            values = std * (max - min) + min
            backtrafo_quantiles.append(values)
            
        # Stack over 3rd axis & return
        return np.stack(backtrafo_quantiles, axis = 2)


class SequenceDataset(torch.utils.data.Dataset):
    """
    Simply takes in the source & target sequences as 3D arrays and returns them as Torch tensors.
    """

    # Store preprocessed source & target sequences
    def __init__(self, source_seq, target_seq): 
        self.source_seq = torch.tensor(source_seq, dtype = torch.float32) # Store source sequences
        self.target_seq = torch.tensor(target_seq, dtype = torch.float32) # Store target sequences
  
    # Return data length  
    def __len__(self):
        return len(self.source_seq) 
  
    # Return a pair of source & target sequences
    def __getitem__(self, idx):
        return self.source_seq[idx], self.target_seq[idx]


class QuantileLoss:
    """
    Takes in targets of shape (...),
    predictions of shape (..., n_quantiles),
    quantiles list.
    
    Returns unreduced quantile loss tensor of shape (..., n_quantiles),
    where each value is quantile loss * 2 (equal to the MAE for q = 0.5).
    
    Implemented from pytorch_forecasting.metrics.quantile.QuantileLoss.
    """

    def __init__(self, quantiles):
        self.quantiles = quantiles

    def loss(self, pred, target):
        
        quantile_losses = []
        for i, q in enumerate(self.quantiles):
            error = target - pred[..., i]
            quantile_error = torch.max(
                (q - 1) * error,
                q * error
            ).unsqueeze(-1)
            quantile_losses.append(quantile_error)

        quantile_losses = torch.cat(quantile_losses, dim = 2)
        return quantile_losses


class LITransformer(L.LightningModule):
    """
    Transformer architecture which takes in inverted sequences of shape (batch_size, features, timesteps) as input.
    
    In both training & inference, the past target values from the source sequence are linearly extrapolated to initialize 
    the target values in the target sequence.
    
    The source sequence consists of past target & covariate values, and is fed to the encoder.
    The target sequence consists of the linear trend prediction & covariate values, and is fed to the decoder.
    No causal masking is applied, as true target values are not fed to the model.
    
    The model outputs quantile predictions.
    """

    # Initialize model
    def __init__(self, hyperparams_dict):
        
        # Delegate function to parent class
        super().__init__() 
        
        # Save external hyperparameters so they are available when loading saved models
        self.save_hyperparameters(logger = False) 

        # Define hyperparameters
        self.source_length = hyperparams_dict["source_length"] # Length of source sequence, L
        self.target_length = hyperparams_dict["target_length"] # Length of target sequence, H
        self.horizon_start = hyperparams_dict["horizon_start"] # Start of the forecast horizon relevant for loss computing
        self.quantiles = hyperparams_dict["quantiles"] # Provide as list of floats: [0.025, 0.5, 0.975]
        self.d_model = hyperparams_dict["d_model"] # Dimensionality of attention inputs, d
        self.n_heads = hyperparams_dict["n_heads"] # N. of attention heads per multiattention block
        self.n_encoders = hyperparams_dict["n_encoders"] # N. of encoder blocks
        self.n_decoders = hyperparams_dict["n_decoders"] # N. of decoder blocks
        self.d_feedforward = hyperparams_dict["d_feedforward"] # Dimensionality of feedforward networks
        self.activation = hyperparams_dict["activation"] # Activation function for transformer FFNs
        self.learning_rate = hyperparams_dict["learning_rate"]
        self.lr_decay = hyperparams_dict["lr_decay"]
        self.dropout_rate = hyperparams_dict["dropout_rate"]

        # Define loss function: Quantile loss
        self.loss = QuantileLoss(quantiles = self.quantiles)
        self.n_quantiles = len(self.quantiles) # Number of quantiles, Q

        # Define architecture components

        # Source projection, input shape (N, D, L), output shape (N, D, d)
        self.source_project = torch.nn.Linear(
            in_features = self.source_length,
            out_features = self.d_model
        )

        # Target projection, input shape (N, D, H), output shape (N, D, d)
        self.target_project = torch.nn.Linear(
            in_features = self.target_length,
            out_features = self.d_model
        )

        # Dropout layer for projections
        self.dropout = torch.nn.Dropout(p = self.dropout_rate)

        # Transformer, input shapes (N, D, d), output shape (N, D, d)
        self.transformer = torch.nn.Transformer(
            d_model = self.d_model,
            nhead = self.n_heads,
            num_encoder_layers = self.n_encoders,
            num_decoder_layers = self.n_decoders,
            dim_feedforward = self.d_feedforward,
            dropout = self.dropout_rate,
            activation = self.activation,
            batch_first = True
        )
        
        # Output layer, input shape (N, D, d), output shape (N, D, H * Q)
        # The output is flattened across timesteps & quantiles, needs to be 
        # indexed & reshaped into shape (N, H, Q) 
        self.output_layer = torch.nn.Linear(
            in_features = self.d_model,
            out_features = self.target_length * self.n_quantiles
        )

    # Define linear trend extrapolation method
    # Input: Past targets(N, L) 
    # Output: Linearly extrapolated future targets(N, H)
    def linear_trend(self, past_target):

        # Get batch size
        batch_size = past_target.shape[0]

        # Get timestep index vectors
        past_idx = torch.arange(0, self.source_length).float()
        future_idx = torch.arange(
            self.source_length, 
            (self.source_length + self.target_length)).float()

        # Get vectors & matrices for linear extrapolation
        ones = torch.ones(self.source_length)
        x = torch.stack((past_idx, ones), 1).to("cpu")
        x_t = torch.transpose(x, 0, 1).to("cpu")
        y = past_target.unsqueeze(-1).to("cpu") 
        
        # Estimate linear extrapolation parameters for each batch
        params = torch.matmul(
            torch.matmul(x_t, y).squeeze(-1), torch.linalg.inv(torch.matmul(x_t, x))
        )
        slopes = params[:, 0]
        constants = params[:, 1]

        # Extrapolate trend to future for each batch
        future_target = future_idx.repeat(batch_size, 1) * slopes.unsqueeze(-1) + constants.unsqueeze(-1)
        return future_target.to(self.device)

    # Define forward propagation
    def forward(self, source_seq, target_seq):

        # Project source & target sequences
        # Inputs: Inverted source(N, D, L) and target(N, D, H) sequences
        source = self.dropout(self.source_project(source_seq))
        target = self.dropout(self.target_project(target_seq))

        # Pass source & target sequences to transformer
        # Causal masking is disabled for the target sequence, as the passed values are predictions
        # Inputs: Projected source and target sequences, (N, D, d)
        transformer_output = self.transformer(
            source, target,
            src_is_causal = False,
            tgt_is_causal = False
        )

        # Pass transformer outputs to output layer, with residual connection to pre-transformer target sequence
        # Input: Transformer output(N, D, d)
        output = self.output_layer(target + transformer_output)

        # Output: Flattened quantile predictions(N, D, H * Q)
        return output

    # Define training step
    # Assumes both source & target sequences are passed to SequenceDataset with D dimensions, target values included
    def training_step(self, batch, batch_idx):

        # Get raw source & target sequences from dataloader
        # Shapes (N, L, D), (N, H, D)
        source_seq, target_seq = batch
        
        # Cast aside real future targets for loss calculation
        real_future_target = target_seq[:, :, 0] # Shape (N, H)
        target_seq = target_seq[:, :, 1:] # Shape (N, H, D-1)

        # Use past targets to extrapolate linear trend to future,
        # concatenate the future trend component with future covariates
        past_target = source_seq[:, :, 0] # Shape (N, L)
        future_target = self.linear_trend(past_target) # Shape (N, H)
        target_seq = torch.cat((
                future_target.unsqueeze(-1),
                target_seq, 
            ), dim = 2) # Shape (N, H, D)

        # Invert the sequences  
        # BRAVO NOLAN
        source_seq = torch.permute(source_seq, (0, 2, 1)) # Shape (N, D, L)
        target_seq = torch.permute(target_seq, (0, 2, 1)) # Shape (N, D, H)

        # Forward propagation
        output = self.forward(source_seq, target_seq) # Shape (N, D, H * Q)

        # Get quantile predictions from output tensor
        preds = output.view(output.shape[0], output.shape[1], self.target_length, self.n_quantiles) # Shape (N, D, H, Q)
        preds = preds[:, 0, :, :] # Shape (N, H, Q)

        # Calculate loss, starting from the chosen future time step  
        loss = self.loss.loss(
            preds[:, self.horizon_start:, :], 
            real_future_target[:, self.horizon_start:]
        ) # Shape (N, H - horizon_start, Q)

        # Reduce the quantile loss
        loss_reduced = loss.mean(dim = 2) # Average over quantiles. Shape (N, H - horizon_start)
        loss_reduced = loss_reduced.sum(dim = 1) # Sum over forecast steps. Shape (N)
        loss_reduced = loss_reduced.mean() # Average over batches. Yields scalar loss for backpropagation.

        # Log the train loss
        self.log("train_loss", loss_reduced, on_step = True, on_epoch = True, prog_bar = True, logger = False)

        return loss_reduced
    
    # Define validation_step
    # Assumes both source & target sequences are passed to SequenceDataset with D dimensions, target values included
    def validation_step(self, batch, batch_idx):

        # Get raw source & target sequences from dataloader
        # Shapes (N, L, D), (N, H, D)
        source_seq, target_seq = batch
        
        # Cast aside real future targets for loss calculation
        real_future_target = target_seq[:, :, 0] # Shape (N, H)
        target_seq = target_seq[:, :, 1:] # Shape (N, H, D-1)

        # Use past targets to extrapolate linear trend to future,
        # concatenate the future trend component with future covariates
        past_target = source_seq[:, :, 0] # Shape (N, L)
        future_target = self.linear_trend(past_target) # Shape (N, H)
        target_seq = torch.cat((
                future_target.unsqueeze(-1),
                target_seq, 
            ), dim = 2) # Shape (N, H, D)

        # Invert the sequences  
        source_seq = torch.permute(source_seq, (0, 2, 1)) # Shape (N, D, L)
        target_seq = torch.permute(target_seq, (0, 2, 1)) # Shape (N, D, H)

        # Forward propagation
        output = self.forward(source_seq, target_seq) # Shape (N, D, H * Q)

        # Get quantile predictions from output tensor
        preds = output.view(output.shape[0], output.shape[1], self.target_length, self.n_quantiles) # Shape (N, D, H, Q)
        preds = preds[:, 0, :, :] # Shape (N, H, Q)

        # Calculate loss, starting from the chosen future time step  
        loss = self.loss.loss(
            preds[:, self.horizon_start:, :], 
            real_future_target[:, self.horizon_start:]
        ) # Shape (N, H - horizon_start, Q)

        # Reduce the quantile loss
        loss_reduced = loss.mean(dim = 2) # Average over quantiles. Shape (N, H - horizon_start)
        loss_reduced = loss_reduced.sum(dim = 1) # Sum over forecast steps. Shape (N)
        loss_reduced = loss_reduced.mean() # Average over batches. Yields scalar loss.

        # Log the validation loss
        self.log("val_loss", loss_reduced, on_step = True, on_epoch = True, prog_bar = True, logger = False)

    # Define prediction_step
    # Assumes target sequences are passed to SequenceDataset with D-1 dimensions, target values excluded
    def predict_step(self, batch, batch_idx):

        # Get raw source & target sequences from dataloader
        # Shapes (N, L, D), (N, H, D-1)
        source_seq, target_seq = batch
        
        # Use past targets to extrapolate linear trend to future,
        # concatenate the future trend component with future covariates
        past_target = source_seq[:, :, 0] # Shape (N, L)
        future_target = self.linear_trend(past_target) # Shape (N, H)
        target_seq = torch.cat((
                future_target.unsqueeze(-1),
                target_seq, 
            ), dim = 2) # Shape (N, H, D)

        # Invert the sequences  
        source_seq = torch.permute(source_seq, (0, 2, 1)) # Shape (N, D, L)
        target_seq = torch.permute(target_seq, (0, 2, 1)) # Shape (N, D, H)

        # Forward propagation
        output = self.forward(source_seq, target_seq) # Shape (N, D, H * Q)

        # Get quantile predictions from output tensor
        preds = output.view(output.shape[0], output.shape[1], self.target_length, self.n_quantiles) # Shape (N, D, H, Q)
        preds = preds[:, 0, :, :] # Shape (N, H, Q)
        return preds

    # Define optimizer & learning rate scheduler
    def configure_optimizers(self):

        # Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        
        # Exponential LR scheduler
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
          optimizer, gamma = self.lr_decay) 
        
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
          "scheduler": lr_scheduler
          }
        }


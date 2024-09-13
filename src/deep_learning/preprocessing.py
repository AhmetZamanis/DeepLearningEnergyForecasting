# Preprocessing steps for the Torch deep learning models
import pandas as pd
import numpy as np
import torch


def get_transformer_sequences(df, input_seq_length = 72, output_seq_length = 32, num_features = 8, horizon_start = 0, forecast_t = 16):
    """
    Takes in the consumption training dataset.
    Returns it as a pair of lists: Input & output sequences, each sequence a dataframe.
    """

    # Get shifted dataset, drop last row due to unknown target
    df_shifted = df.copy()
    df_shifted["consumption_MWh"] = df_shifted.consumption_MWh.shift(-1)
    df_shifted = df_shifted.dropna()
    n_steps = len(df_shifted) 

    # Find the index of the first row in the data at hour T, where the index is bigger than source_length - 1. This will be the first T.
    first_t = df_shifted.loc[(df_shifted.time.dt.hour == forecast_t) & (df_shifted.index >= input_seq_length - 1)].index.values[0]

    # Find the index of the last row in the data at hour T, with `output_seq_length` time steps after it. This will be the last T.
    last_t = df_shifted.loc[(df_shifted.time.dt.hour == forecast_t) & (df_shifted.index + output_seq_length - 1 <= df.index.values[-1])].index.values[-2]

    # Number of T rows followed by a sufficient length input & output sequence
    n_sequences = (last_t - first_t) // 24 + 1 

    # Initialize lists of sequences
    input_sequences = []
    output_sequences = []
    
    # Get sequences
    for t in range(first_t, last_t + 1, 24):

        # Get input sequence
        new_input = pd.concat([
            df_shifted.iloc[(t - input_seq_length):t, 0], # Time
            df_shifted.iloc[(t - input_seq_length):t, 2], # Past target
            df_shifted.iloc[(t - input_seq_length):t, 3:] # Past covariates
            ], axis = 1)
        new_input = new_input.set_index("time")
    
        # Get output sequence
        new_output = pd.concat([
            df_shifted.iloc[t:(t + output_seq_length), 0], # Time 
            df_shifted.iloc[t:(t + output_seq_length), 1], # Future target
            df_shifted.iloc[t:(t + output_seq_length), 3:] # Future known covariates
            ], axis = 1)
        new_output = new_output.set_index("time")
    
        # Concatenate to arrays of sequences
        input_sequences.append(new_input)
        output_sequences.append(new_output)

    return input_sequences, output_sequences


class SequenceScaler:
    """
    Takes in 1 pair of lists: Input & output sequences, each sequence a dataframe.

    Returns scaled 3D numpy array of shape (observations, timesteps, features).
    Can also backtransform scaled predictions.
    """

    def __init__(self, feature_range = (-1, 1)):
        self.lower = feature_range[0]
        self.upper = feature_range[1]

    def fit(self, input_df, output_df):

        # Get input & output sequences as 3D arrays
        # The time index will be skipped, yielding shape (N, seq_length, seq_dims)
        input = np.stack(input_df, axis = 0)
        output = np.stack(output_df, axis = 0)

        # Get number of dimensions
        self.num_dimensions = input.shape[2]
        
        # Extract & save minimum, maximum for each dimension
        dimensions_mini = []
        dimensions_maxi = []
        for dimension in range(0, self.num_dimensions):
            min = np.min([
                np.min(input[:, :, dimension]),
                np.min(output[:, :, dimension])
            ])
            dimensions_mini.append(min)

            max = np.max([
                np.max(input[:, :, dimension]),
                np.max(output[:, :, dimension])
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
    Simply takes in the input & output sequences as 3D arrays and returns them as Torch tensors.
    """

    # Store preprocessed input & output sequences
    def __init__(self, input_seq, output_seq): 
        self.input_seq = torch.tensor(input_seq, dtype = torch.float32) # Store input sequences
        self.output_seq = torch.tensor(output_seq, dtype = torch.float32) # Store output sequences
  
    # Return data length  
    def __len__(self):
        return len(self.input_seq) 
  
    # Return a pair of input & output sequences
    def __getitem__(self, idx):
        return self.input_seq[idx], self.output_seq[idx]

# Performance testing for Torch deep learning models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error as mae, mean_absolute_percentage_error as mape, root_mean_squared_log_error as rmsle, mean_pinball_loss as pinball


def train_val_split(input_sequences, output_sequences, train_fraction = 0.8, batch_size = 64):
    """
    Takes in 1 pair of lists: Input & output sequences, each sequence a dataframe.
    Returns 2 pairs of lists: Input & output sequences, split into training & validation sets.
    """

    # Get the index of the last training sequence, get training set
    train_end = int(len(input_sequences) * train_fraction)
    train_input_sequences, train_output_sequences = input_sequences[0:train_end], output_sequences[0:train_end]

    # Trim the start of the training set so batch size is constant
    train_remainder = len(train_input_sequences) % batch_size
    train_input_sequences, train_output_sequences = train_input_sequences[train_remainder:], train_output_sequences[train_remainder:]

    # Get validation set
    val_input_sequences, val_output_sequences = input_sequences[train_end:], output_sequences[train_end:]
    val_remainder = len(val_input_sequences) % batch_size
    val_input_sequences, val_output_sequences = val_input_sequences[:-val_remainder], val_output_sequences[:-val_remainder]

    return train_input_sequences, train_output_sequences, val_input_sequences, val_output_sequences
    

# def train_val_test_split(input_sequences, output_sequences, train_fraction = 0.6, val_fraction = 0.2, batch_size = 64):
#     """
#     Takes in 1 pair of lists: Input & output sequences, each sequence a dataframe.
#     Returns 3 pairs of lists: Input & output sequences, split into training, validation & testing sets.
#     """

#     # Get the index of the last training sequence, get training set
#     train_end = int(len(input_sequences) * train_fraction)
#     train_input_sequences, train_output_sequences = input_sequences[0:train_end], output_sequences[0:train_end]

#     # Trim the start of the training set so batch size is constant
#     train_remainder = len(train_input_sequences) % batch_size
#     train_input_sequences, train_output_sequences = train_input_sequences[train_remainder:], train_output_sequences[train_remainder:]

#     # Get validation set
#     val_end = train_end + int(len(input_sequences) * val_fraction)
#     val_input_sequences, val_output_sequences = input_sequences[train_end:val_end], output_sequences[train_end:val_end]
#     val_remainder = len(val_input_sequences) % batch_size
#     val_input_sequences, val_output_sequences = val_input_sequences[:-val_remainder], val_output_sequences[:-val_remainder]

#     # Get testing set
#     test_input_sequences, test_output_sequences = input_sequences[val_end:], output_sequences[val_end:]
#     test_remainder = len(test_input_sequences) % batch_size
#     test_input_sequences, test_output_sequences = test_input_sequences[:-test_remainder], test_output_sequences[:-test_remainder]

#     ### Have to split training data @ validation and training data @ testing step separately, to avoid losing sequences for the latter
#     ### Do it with one function: First train-test split, then split train into train-val.

#     return train_input_sequences, val_input_sequences, test_input_sequences, train_output_sequences, val_output_sequences, test_output_sequences

def test_sequences_to_dataframe(test_input_seq, test_output_seq):
    """
    Takes in 1 pair of lists: Input & output sequences for the testing set.
    Returns a dataframe of testing set observations.
    """

    # Create dataframe from all output sequences
    test_output_dates = np.stack([sequence.index for sequence in test_output_seq], axis = 0)  # Is this really necessary? Not sure, but wouldn't hurt
    test_output_stacked = np.stack(test_output_seq, axis = 0)
    df_test_output = pd.DataFrame({
        "time": np.ravel(test_output_dates),
        "consumption_MWh": np.ravel(test_output_stacked[:, :, 0])
    })
    df_test_output["time"] = pd.to_datetime(df_test_output["time"])
    df_test_output["sequence"] = "output"
    
    # Create dataframe from all input sequences
    test_input_dates = np.stack([sequence.index for sequence in test_input_seq], axis = 0)
    test_input_stacked = np.stack(test_input_seq, axis = 0)
    df_test_input = pd.DataFrame({
        "time": np.ravel(test_input_dates),
        "consumption_MWh" : np.ravel(test_input_stacked[:, :, 0])
    })
    df_test_input["time"] = pd.to_datetime(df_test_input["time"])
    df_test_input["sequence"] = "input"
    
    # Concatenate & sort by time
    df_test = pd.concat([df_test_output, df_test_input])
    df_test = df_test.sort_values("time")

    return df_test


def plot_actual_predicted(df_test, df_preds, model, quantile_interval = "95", ax = None):
    """
    Takes in dataframes of the testing data & model quantile predictions.
    Plots the actual vs. predicted plot, prediction intervals, for entire testing set.
    """

    if ax == None:
        fig, ax = plt.subplots()

    _ = sns.lineplot(
        data = df_test,
        x = "time",
        y = "consumption_MWh",
        label = "Actual values",
        ax = ax
    )

    _ = sns.lineplot(
        data = df_preds,
        x = "time",
        y = "pred_point",
        label = f"Predictions, {quantile_interval}% quantile interval",
        ax = ax
    )
    
    _ = ax.fill_between(
        x = df_preds.time,
        y1 = df_preds.pred_low,
        y2 = df_preds.pred_high,
        label = f"{quantile_interval}% prediction interval",
        color = "orange",
        alpha = 0.4
    )
    _ = ax.set_title(f"Model: {model}")

    return ax


def plot_sequence_preds(preds_array, test_input_seq, test_output_seq, model, sequence_index = 0, quantile_interval = "95", ax = None):
    """
    Takes in: 
        - Array of model quantile predictions,
        - 1 pair of lists: Input & output sequences for the testing set,
        - Index of testing input & output sequence pair to be plotted.
        
    Plots the actual vs. predicted plot, prediction intervals, for selected input & output sequence pair.
    """

    # Get n. of sequences
    n_sequences = len(test_output_seq)
    
    # Get predictions for selected sequence
    preds_low = preds_array[sequence_index, :, 0]
    preds_point = preds_array[sequence_index, :, 1]
    preds_high = preds_array[sequence_index, :, -1]

    # Get & combine actual outputs, inputs, dates
    date_output = test_output_seq[sequence_index].index.to_series()
    output = test_output_seq[sequence_index].consumption_MWh.values
    
    date_input = test_input_seq[sequence_index].index.to_series()
    input_vals = test_input_seq[sequence_index].consumption_lag2.values

    date = pd.concat([date_input, date_output], axis = 0)
    actual = np.concatenate([input_vals, output], axis = 0)

    if ax == None:
        fig, ax = plt.subplots()

    # Plot
    _ = sns.lineplot(
        x = date,
        y = actual,
        label = "Actual values",
        ax = ax
    )

    _ = sns.lineplot(
        x = date_output,
        y = preds_point,
        label = f"Predictions, {quantile_interval}% quantile interval",
        ax = ax
    )
    
    _ = ax.fill_between(
        x = date_output,
        y1 = preds_low,
        y2 = preds_high,
        color = "orange",
        alpha = 0.4,
        label = f"Predictions, {quantile_interval}% quantile interval"
    )

    _ = ax.set_title(f"Model: {model},\n Sequence index: {sequence_index} of {n_sequences - 1}")
    _ = ax.set_xlabel("time")
    _ = ax.set_ylabel("consumption")

    return ax


def calculate_metrics(df_test, df_preds, model, quantiles = [.025, 0.5, .975], rounding = 4):
    """
    Takes in dataframes of the testing data & model quantile predictions.
    Returns a dataframe of performance metrics.
    """

    # Select output sequences only
    df_test = df_test[df_test["sequence"] == "output"]

    df_metrics = pd.DataFrame([
        mape(df_test.consumption_MWh, df_preds.pred_point) * 100,
        rmsle(df_test.consumption_MWh, df_preds.pred_point),
        mae(df_test.consumption_MWh, df_preds.pred_point),
        pinball(df_test.consumption_MWh, df_preds.pred_low, alpha = quantiles[0]),
        pinball(df_test.consumption_MWh, df_preds.pred_point, alpha = quantiles[1]),
        pinball(df_test.consumption_MWh, df_preds.pred_high, alpha = quantiles[-1])
    ], columns = [f"Model: {model}"],
    index = [ 
        "MAPE, point", 
        "RMSLE, point",
        "MAE, point",
        f"Pinball loss, q: {quantiles[0] * 100}%",
        f"Pinball loss, q: {quantiles[1] * 100}%",
        f"Pinball loss, q: {quantiles[-1] * 100}%"
    ]).round(rounding)

    return df_metrics
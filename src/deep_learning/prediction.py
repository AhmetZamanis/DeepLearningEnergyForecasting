# Handling & reformatting the predictions of Torch deep learning models
import pandas as pd
import numpy as np


def get_prediction_data(df, input_seq_length = 72, output_seq_length = 32, num_features = 8):
    """
    Takes in the consumption training dataset.
    Returns a dataset with the last `input_seq_length` training observations, 
    plus `output_seq_length` prediction observations.
    """

    # Remember to initialize target values as None's

    # Take T as a variable? Or shift the data & extract here the last L steps as prediction input seq, create next H steps as prediction output seq, circumventing get_transformer_sequences?


def predictions_to_dataframe(preds_array, preds_output_seq):
    """
    Takes in 3D numpy array of quantile predictions, the prediction target sequence(s) as a list of dataframes.
    Returns a dataframe of quantile predictions & their respective dates.
    """
    
    test_dates = np.stack([sequence.index for sequence in preds_output_seq], axis = 0)  # WILL THIS WORK WITH 1 TARGET SEQUENCE?
    df_preds = pd.DataFrame({
        "time": np.ravel(test_dates),
        "pred_low": np.ravel(preds_array[:, :, 0]),
        "pred_point": np.ravel(preds_array[:, :, 1]),
        "pred_high": np.ravel(preds_array[:, :, -1])
    })

    return df_preds
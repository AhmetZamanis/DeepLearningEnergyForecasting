# Handling & reformatting the predictions of Torch deep learning models
import pandas as pd
import numpy as np


def predictions_to_dataframe(preds_array, preds_output_seq):
    """
    Takes in 3D numpy array of quantile predictions, the prediction target sequence(s) as a list of dataframes.
    Returns a dataframe of quantile predictions & their respective dates.
    """
    
    test_dates = np.stack([sequence.index for sequence in preds_output_seq], axis = 0)
    df_preds = pd.DataFrame({
        "date": np.ravel(test_dates),
        "pred_low": np.ravel(preds_array[:, :, 0]),
        "pred_point": np.ravel(preds_array[:, :, 1]),
        "pred_high": np.ravel(preds_array[:, :, -1])
    })

    return df_preds
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics import tsaplots


def plot_autocorrelation(df, df_daily, variable = "USD/MWh", lags_hourly = 48, lags_daily = 30):

    fig, ax = plt.subplots(2, 2)

    # Autocorrelation, hourly
    tsaplots.plot_acf(
        ax = ax[0, 0],
        x = df[variable],
        lags = lags_hourly,
        title = "ACF hourly, " + variable
    )

    # Partial autocorrelation, hourly
    tsaplots.plot_pacf(
        ax = ax[1, 0],
        x = df[variable],
        lags = lags_hourly,
        title = "PACF hourly, " + variable
    )

    # Autocorrelation, daily
    tsaplots.plot_acf(
        ax = ax[0, 1],
        x = df_daily[variable],
        lags = lags_daily,
        title = "ACF daily, " + variable
    )

    # Partial autocorrelation, daily
    tsaplots.plot_pacf(
        ax = ax[1, 1],
        x = df_daily[variable],
        lags = lags_daily,
        title = "PACF daily, " + variable
    )
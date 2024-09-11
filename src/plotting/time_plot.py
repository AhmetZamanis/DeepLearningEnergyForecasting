import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def time_plot(df, df_daily, variable = "consumption_MWh", horizontal_color = "crimson"):

    # Process means for horizontal lines
    mean_hourly = df[variable].mean()
    mean_daily = df_daily[variable].mean()

    # Figure
    fig, ax = plt.subplots(2)
    fig.suptitle(variable)

    # Hourly plot
    _ = sns.lineplot(
    data = df,
    x = "time",
    y = variable,
    ax = ax[0]
    )
    _ = ax[0].set_xlabel("hourly")
    _ = ax[0].axhline(y = mean_hourly, c = horizontal_color, label = "mean")
    _ = ax[0].legend()
    
    # Daily plot
    _ = sns.lineplot(
    data = df_daily,
    x = "time",
    y = variable,
    ax = ax[1]
    )
    _ = ax[1].set_xlabel("daily aggregation")
    _ = ax[1].axhline(y = mean_daily, c = horizontal_color)
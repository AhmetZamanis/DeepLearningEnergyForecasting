import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def seasonal_plot1(df, variable = "consumption_MWh", groupby_year = True):

    # Grouping variable
    hue = None
    if groupby_year:
        hue = df.time.dt.year.astype(str)

    fig, ax = plt.subplots(2, 2)
    fig.suptitle("Seasonality, " + variable)

    # Hour of day
    _ = sns.lineplot(
        x = df.time.dt.hour,
        y = df[variable],
        hue = hue,
        marker = "o",
        markersize = 4,
        legend = False,
        ax = ax[0, 0]
    )
    _ = ax[0, 0].set_xticks(range(0, 25, 6))
    _ = ax[0, 0].set_xlabel("hour of day")

    # Day of week
    _ = sns.lineplot(
        x = df.time.dt.weekday,
        hue = hue,
        marker = "o",
        markersize = 4,
        legend = False,
        y = df[variable],
        ax = ax[0, 1]
    )
    _ = ax[0, 1].set_xticks(range(0, 7, 1))
    _ = ax[0, 1].set_xlabel("day of week, monday-sunday")

    # Week of year
    _ = sns.lineplot(
        x = df.time.dt.isocalendar().week,
        hue = hue,
        legend = False,
        y = df[variable],
        ax = ax[1, 0]
    )
    _ = ax[1, 0].set_xticks(range(0, 54, 8))
    _ = ax[1, 0].set_xlabel("week of year")

    # Month of year
    _ = sns.lineplot(
        x = df.time.dt.month,
        y = df[variable],
        hue = hue,
        marker = "o",
        markersize = 4,
        ax = ax[1, 1]
    )
    _ = ax[1, 1].set_xticks(range(1, 13, 1))
    _ = ax[1, 1].set_xlabel("month")

    if not groupby_year:
        return
    
    _ = ax[1, 1].legend(
            title = "year", 
            bbox_to_anchor = (1.05, 1.0), 
            fontsize = "small",
            loc = "best"
    )


def seasonal_plot2(df, variable = "consumption_MWh", groupby_month = True):

    # Grouping variable
    hue = None
    if groupby_month:
        hue = df.time.dt.month.astype(str)

    fig, ax = plt.subplots(2)
    fig.suptitle("Seasonality, " + variable)

    # Hour of day
    _ = sns.lineplot(
        x = df.time.dt.hour,
        y = df[variable],
        hue = hue,
        markers = hue,
        #marker = "o",
        markersize = 4,
        ax = ax[0]
    )
    _ = ax[0].set_xticks(range(0, 25, 6))
    _ = ax[0].set_xlabel("hour of day")

    # Day of week
    _ = sns.lineplot(
        x = df.time.dt.weekday,
        hue = hue,
        markers = hue,
        #marker = "o",
        markersize = 4,
        legend = False,
        y = df[variable],
        ax = ax[1]
    )
    _ = ax[1].set_xticks(range(0, 7, 1))
    _ = ax[1].set_xlabel("day of week, monday-sunday")

    if not groupby_month:
        return
    
    _ = ax[0].legend(
            title = "month", 
            bbox_to_anchor = (1.05, 1.0), 
            fontsize = "small",
            loc = "best"
    )


def seasonal_plot3(df, variable = "consumption_MWh"):

    hue = df.time.dt.month.astype(str)

    fig, ax = plt.subplots(2)
    fig.suptitle("Seasonality, " + variable)

    # Day of month, ungrouped
    _ = sns.lineplot(
        x = df.time.dt.day,
        y = df[variable],
        hue = None,
        marker = "o",
        markersize = 4,
        legend = False,
        ax = ax[0]
    )
    _ = ax[0].set_xticks(range(1, 32, 6))
    _ = ax[0].set_xlabel("day of month")

    # Day of month, grouped
    _ = sns.lineplot(
        x = df.time.dt.day,
        y = df[variable],
        hue = hue,
        ax = ax[1]
    )
    _ = ax[1].set_xticks(range(1, 32, 6))
    _ = ax[1].set_xlabel("day of month")

    _ = ax[1].legend(
            title = "month", 
            bbox_to_anchor = (1.05, 1.0), 
            fontsize = "small",
            loc = "best"
    )
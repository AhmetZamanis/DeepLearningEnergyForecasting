## Introduction
This report summarizes my experience experimenting with some deep learning forecasting models, on a time series dataset of hourly energy consumption values.
I used PyTorch & Lightning to build & benchmark an LSTM-based and a Transformer-based forecasting model. The goal was to familiarize & experiment with these architectures, and better understand some variants developed for time series forecasting.
\
\
This report will go over the problem formulation, the architectures of the models built, and their performance comparisons. We'll also talk about some papers & implementations that inspired this experiment. The code & implementation details can be found in Jupyter notebooks in this repository.
### Data overview
The data consists of hourly, country-wide energy consumption values in Türkiye. The date range spans across five years, from January 1st 2018 to December 31st 2023. 
\
\
I added some time covariates: A trend dummy, and cyclical covariates based on the hour, day of week, and the month. These were used alongside the past consumption values as features in the models. All variables including the target values were scaled from -1 to 1.
![Data](https://github.com/AhmetZamanis/DeepLearningEnergyForecasting/blob/main/ReportImages/DataHead.png)

### Problem formulation
I based my problem formulation on the operatio
### Stateful LSTM model
### Linear inverted transformer model
### Performance comparison
### Sources & acknowledgements
The data was sourced by myself, from the EPİAŞ Transparency Platform ([website link](https://seffaflik.epias.com.tr/home)), which provides open-access data on Türkiye's energy market. The website & API is available in English, though access to the API requires an access application to be made using a static IP address. 

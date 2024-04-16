This repository contains the code & results of a multi-step time series forecasting exercise I performed with deep learning models, on a large dataset of hourly energy consumption values. 

I used PyTorch Lightning to implement a stateful LSTM model, and an inverted Transformer model, with some modifications inspired by multiple other time series forecasting architectures. Most notably, I implemented a simple linear extrapolation method in the Transformer model, as a simple way to initialize target variable values for the decoder target sequence.

See the [Markdown report](https://github.com/AhmetZamanis/DeepLearningEnergyForecasting/blob/main/Report.md) for an explanation of the models & results, along with sources & acknowledgements.

I also used this dataset and the GPyTorch package to try out Gaussian Process Regression with various training strategies. See the [notebook](https://github.com/AhmetZamanis/DeepLearningEnergyForecasting/blob/main/4.0_GaussianProcess.ipynb) for details. 

The dataset is available on [Kaggle](https://www.kaggle.com/datasets/ahmetzamanis/energy-consumption-and-pricing-trkiye-2018-2023).

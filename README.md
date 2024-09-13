This repository contains the code & results of a multi-step time series forecasting exercise I performed with deep learning models, on a large dataset of hourly energy consumption values. 

I used `pytorch` & `lightning` to implement a stateful LSTM model, and an inverted Transformer model, with some modifications inspired by multiple other time series forecasting architectures. Most notably, I implemented a simple linear extrapolation method in the Transformer model, as a simple way to initialize target variable values for the decoder target sequence.

See `Report.md` for an explanation of the models & results, along with sources & acknowledgements. See `analysis` for the data prep, EDA & modeling notebooks, and `src` for methods including Torch classes.

I also used this dataset and the `GPyTorch` package to try out Gaussian Process Regression with various training strategies. See `analysis` for the notebook. 

The merged dataset is available on [Kaggle](https://www.kaggle.com/datasets/ahmetzamanis/energy-consumption-and-pricing-trkiye-2018-2023).

`Python version: 3.12.2`

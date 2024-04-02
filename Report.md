## Introduction
This report summarizes my experience experimenting with some deep learning forecasting models, on a time series dataset of hourly energy consumption values.
I used PyTorch and Lightning to build and benchmark an LSTM-based and a Transformer-based forecasting model. The goal was to familiarize and experiment with these architectures, and better understand some variants developed for time series forecasting.
\
\
This report will go over the problem formulation, the architectures of the models built, and their performance comparisons. We'll also talk about some architectures that inspired this experiment. The code & implementation details can be found in Jupyter notebooks in this repository. The Torch classes are also available in Python scripts, but keep in mind they are restrictive in terms of the types of covariates they support.
### Data overview
The data [[1]](#) consists of hourly, country-wide energy consumption values in Türkiye. The date range spans across five years, from January 1, 2018 to December 31, 2023. 
\
\
I added some time covariates: A trend dummy, and cyclical encoded covariates based on the hour, weekday and the month. These were used alongside the past consumption values as features in the models. All variables including the target values were scaled from -1 to 1, leaving the cyclical covariates unchanged.
![Data](https://github.com/AhmetZamanis/DeepLearningEnergyForecasting/blob/main/ReportImages/DataHead.png)
\
\
As expected, the energy consumption displays strong seasonality, mainly hourly, but also by weekday and possibly monthly seasonality. There is also a slight linear increase trend across the data span, and some potential cyclicality. For more detail, see the data processing, EDA & feature engineering notebooks.
### Problem formulation
I based my problem formulation on the operations of the EPİAŞ energy market in Türkiye. 
- Essentially, every day before 17:00, all energy producers have to commit hourly energy production amounts for tomorrow.
- Based on this, I took 16:00 every day as the forecast start point, and built models that forecast the next 32 time steps: The hourly consumption values for the remaining 8 hours of this day, and the hourly consumption values for the following 24 hours.
- Only the following 24 hours would likely be of interest in a real-life scenario, but I decided to set the forecast horizon to 32 to avoid gaps.
  - The code offers the option to exclude the first N forecasts from the models' loss calculations, though I am unsure if this is a good idea, especially for the LSTM model.
- Since the hourly consumption data is available with a 2-hour lag, at time T, we would only have access to the consumption values for T-2 and backwards. The models accordingly use consumption at T-2 as the past target value at T.

Deep learning forecasting models process time series data as sequences: For this problem, I used past sequences of 72 hours as the lookback window for each forecast start point, and aimed to predict future sequences of 32 hours.
### Stateful LSTM model
The first model I used is a recurrent architecture, based on one or multiple long-short-term memory layers.
- LSTM layers process sequence data step by step. In one pass, the input values for one timestep are passed through the network. For our data, that is 8 network inputs.
- The outputs (hidden & cell states) of an LSTM layer at timestep T are passed as inputs to the LSTM layer at timestep T+1, along with the input values for timestep T+1.
- This way, the LSTM architecture aims to learn temporal dependencies, essentially storing them in "memory" cells.

One nuance with this architecture is the initialization of hidden & cell states.
- Typically, the hidden & cell state values for the first timestep in each training batch are initialized as zeroes.
- This approach results in a **stateless LSTM**: The hidden & cell state is essentially reset with every batch, and we rely only on the model parameters to capture temporal dependencies. Memory is not retained across batches.
- In contrast, the **stateful LSTM** approach uses the last hidden & cell states for observation N in batch 1 as the initial hidden & cell states for observation N in batch 2. Memory is retained across batches [[2]](#).

The stateless architecture is easier to implement, and it is likely sufficient for many problems, but I implemented a stateful architecture out of curiosity.
- The model I implemented retains the last hidden & cell states from the previous training batch & passes them on to the next training batch.
  - The first hidden & cell states in each training epoch are initialized as zeroes.   
- This holds across training, validation & inference, which may be useful to capture & propagate long-term dependencies, but may also perform badly with time gaps between training & inference sets.
  - In this example, there is no gap between the training, validation & testing sets, so we propagate the hidden & cell states from the start of the data to the end.
 
Another nuance is that the model actually forecasts one timestep in one network pass, as it needs a hidden & cell state in each timestep.
- To get predictions for 32 timesteps, we essentially make 32 passes to the network, in each pass shifting the 72-step input sequence forward by one step.
- During training, we use the known target values to extend the input sequence into the next timestep.
  - This is similar to "teacher forcing" in language models. 
- During validation & inference, we use the prediction from the previous step to extend the input sequence into the next timestep.
  - Essentially, we are making predictions based on predictions after the first target timestep, so errors will stack.

Besides these modifications, the model is essentially a multi-layer LSTM block, similar to DeepAR [[3]](#).
- Unlike DeepAR, I opted to use a quantile loss function to generate quantile forecast intervals, from a linear output layer with one output per forecasted quantile.
- By default, I forecasted the 2.5th, 50th and 97.5th quantiles, essentially a median forecast surrounded by a 95% interval.

I tuned model hyperparameters with Optuna, and the best performing tune is fairly simple, with only 1 LSTM layer, a hidden size of 8 equal to the input size, and only 7 training epochs to get the best model iteration. A small amount of dropout and exponential learning rate scheduling was also used.
### Linear inverted transformer model
### Performance comparison
### Sources & acknowledgements
<a id="1">[1]<a/> The data was sourced by myself, from the EPİAŞ Transparency Platform , which provides open-access data on Türkiye's energy market. The website & API are available in English, though access to the API requires an access application to be made using a static IP address. [Website link](https://seffaflik.epias.com.tr/home)
\
\
<a id="2">[2]<a/> A useful article with visualizations that helped me better understand how the stateful LSTM propagates states across batches. [Article link](https://towardsai.net/p/l/stateless-vs-stateful-lstms)
\
\
<a id="2">[3]<a/> D. Salinas, V. Flunkert, J. Gasthaus, DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks, (2019) [arXiv:1704.04110](https://arxiv.org/abs/1704.04110)

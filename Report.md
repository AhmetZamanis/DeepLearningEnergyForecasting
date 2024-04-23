## Introduction
This report summarizes my experience experimenting with some deep learning forecasting models, on a time series dataset of hourly energy consumption values.
I used PyTorch and Lightning to build and benchmark two multi-horizon forecasting models: An LSTM-based model, and a Transformer-based model. The goal was to familiarize and experiment with these architectures, and better understand some variants developed for time series forecasting.
\
\
This report will go over the problem formulation, the architectures of the models built, and their performance comparisons. We'll also talk about some architectures that inspired this experiment. The code & implementation details can be found in Jupyter notebooks in this repository. The Torch classes are also available in Python scripts, but keep in mind the implementations are tailored to this problem & dataset.

### Data overview
The data [[1]](#1) consists of hourly, country-wide energy consumption values in Türkiye. The date range spans across five years, from January 1, 2018 to December 31, 2023. 
\
\
I added some time covariates: A trend dummy, and cyclical encoded covariates based on the hour, weekday and the month. These were used alongside the past consumption values as features in the models. All variables including the target values were scaled from -1 to 1, leaving the cyclical covariates unchanged.
![Data](https://github.com/AhmetZamanis/DeepLearningEnergyForecasting/blob/main/ReportImages/DataHead.png)
\
\
As expected, the energy consumption displays strong seasonality, mainly hourly, but also by weekday and possibly monthly seasonality. There is also a slight linear increase trend across the data span, and some potential cyclicality. For more detail, see the data processing, EDA & feature engineering notebooks.
\
\
I also experimented with using multiple Fourier terms chosen according to the strongest seasonality frequencies in the consumption time series, instead of cyclical encoding seasonal features. The mathematics behind the two approaches is fairly similar, and so were the results. See the relevant [branch](https://github.com/AhmetZamanis/DeepLearningEnergyForecasting/tree/add-fourier-feature) for that version of the feature engineering & testing notebooks.

### Problem formulation
I based my problem formulation on the operations of the EPİAŞ energy market in Türkiye. 
- Essentially, every day before 17:00, all energy producers have to commit hourly energy production amounts for the next day.
- Based on this, I took 16:00 every day as the forecast start point, and built models that forecast the next 32 time steps: The hourly consumption values for the remaining 8 hours of this day, and the hourly consumption values for the following 24 hours.
- Only the following 24 hours would likely be of interest in a real-life scenario, but I decided to set the forecast horizon to 32 to avoid gaps.
  - The code offers the option to exclude the first N forecasts from the models' loss calculations, though I am unsure if this is a good idea, especially for the LSTM model.
- Since the hourly consumption data is available with a 2-hour lag, at time T, we would only have access to the consumption values for T-2 and backwards. The models accordingly use consumption at T-2 as the past target value at T.

Deep learning forecasting models process time series data as sequences: For this problem, I used past sequences of 72 hours as the lookback window for each forecast start point, and aimed to predict future sequences of 32 hours.

### Stateful LSTM model
The first model I used is a recurrent architecture, based on one or multiple long-short-term memory layers.
- LSTM layers process sequence data step by step. In one pass, the input values for one timestep are passed through the network. For our data, that is 8 network inputs.
- The outputs (hidden & cell states) of an LSTM layer at timestep T are passed as inputs to the LSTM layer at timestep T+1, along with the input values for timestep T+1.
- This way, the LSTM architecture aims to learn temporal dependencies, essentially retaining past information in "memory" cells.

One nuance with this architecture is the initialization of hidden & cell states.
- Typically, the hidden & cell state values for the first timestep in each training batch are initialized as zeroes.
- This approach results in a **stateless LSTM**: The hidden & cell state is essentially reset with every batch, and we rely only on the model parameters to capture temporal dependencies. We assume the batches are independent, and the lookback window captures all relevant history about the process. Memory is not retained across batches.
- In contrast, the **stateful LSTM** approach uses the last hidden & cell states for observation N in batch 1 as the initial hidden & cell states for observation N in batch 2. Memory is retained across batches [[2]](#2).

The stateless architecture is easier to implement and train, and it is likely sufficient for many problems, but I implemented a stateful architecture out of curiosity.
- The model I implemented retains the last hidden & cell states from the previous batch & passes them on to the next batch.
  - The first hidden & cell states in each training epoch are initialized as zeroes.   
- The hidden & cell states are propagated across training, validation & inference batches, which may be useful to capture long-term dependencies, but may also perform badly with time gaps between training & inference sets.
  - In this example, there is no gap between the training, validation & testing sets, so we propagate the hidden & cell states from the start of the data to the end. See notebook 3.1 for the implementation details on this.
 
Another nuance is that the model actually forecasts one timestep in one network pass, as it needs a hidden & cell state in each timestep.
- To get predictions for 32 timesteps, we essentially make 32 passes to the network, in each pass shifting the 72-step input sequence forward by one step.
- During training, we use the known target values to extend the input sequence into the next timestep, as training on predictions may mislead the model's learning.
  - This is similar to **teacher forcing** in language models. 
- During validation & inference, we use the prediction from the previous step to extend the input sequence into the next timestep.
  - Essentially, we are making predictions based on predictions after the first target timestep, so errors will stack.

Besides these modifications, the model is essentially a multi-layer LSTM block, similar to **DeepAR** [[3]](#3).
- Unlike DeepAR, I opted to use a quantile loss function to generate quantile forecast intervals, from a linear output layer with one output per forecasted quantile.
- By default, I forecasted the 2.5th, 50th and 97.5th quantiles, essentially a median forecast surrounded by a 95% interval.
- The training loss is calculated by averaging over the quantiles, summing over the timesteps, and averaging over the batches.
  - There is room for experimentation here: For example, if particular timesteps' forecasts are more important, we could take the weighted average of losses over timesteps and put more weight on more important timesteps, instead of summing them all up and treating them equally.

I tuned model hyperparameters with Optuna, and the best performing tunes are fairly simple, usually with only 1 LSTM layer, a hidden size of 8 equal to the input size, and around 10 training epochs to get the best model iteration. A small amount of dropout was also used, along with exponential learning rate scheduling.

### Linear inverted transformer model
The second model I used is a **Transformer architecture** [[4]](#4), which is best known for its use in language tasks. I made some modifications for time series forecasting. 
- The transformer architecture consists of an encoder & decoder block, which run in parallel.
- Typically, a "source" sequence that represents the input is fed to the encoder, and a "target" sequence that represents the desired output is fed to the decoder.
- Transformers utilize the **self-attention mechanism** [[5]](#5) to process all sequence steps in one network pass, instead of handling them sequentially like recurrent architectures.
- A key benefit is computational efficiency. The data handling & model logic is also simpler, with the ability to generate multi-step predictions in one go, without the need for hidden states from previous time steps.

Many architectures that aim to adapt the Transformer model to time series forecasting exist. Out of these, a key inspiration for my implementation is the recently published **iTransformer** (Inverted Transformer) [[6]](#6).
- In a default Transformer, The sequences of shape (timesteps, features) are projected to a fixed size across the features dimension, yielding (timesteps, dimensions). Then, attention is applied to each feature value at a given timestep.
- The iTransformer inverts the sequences into shape (features, timesteps), and projects them to a fixed size across the timesteps dimension. This way, attention is applied to each timestep value for a given feature.
- The authors suggest this is more suitable to learn relationships across timesteps, which intuitively made sense to me. Therefore, the model I implemented also inverts the source & target sequences before passing them to the network.
- Transformers often apply positional encodings to the input sequences to capture the ordering of sequence steps, as they are not processed sequentially. The iTransformer does not apply positional encodings: The authors state the feed-forward networks applied to inverted inputs inherently capture the sequence orderings. Positional encodings were also omitted from my model.

Unlike the iTransformer, which is an encoder-only model, the Transformer model in this repository employs a typical encoder-decoder Transformer.
- From what I understand, the iTransformer, or at least its first version, only takes in the past values & covariates as a "source" sequence. It does not natively support future covariate values for the forecast horizon.
  - There could be a workaround that I am not aware of.
- I instead opted to use both a source & target sequence, attended by the encoder & decoder separately, as the seasonal future covariates are likely to be critical for forecasting this highly seasonal time series.
- This architecture should natively support source & target sequences with different numbers of features, so any number of past & future covariates can be used, though in this experiment the source & target sequences represent the same features across different sequence lengths.
  - I haven't considered how static covariates could be supported, since the time series in question is univariate. A simple but likely inefficient method is concatenating them to the sequences over the feature dimension, with a fixed value across all timesteps.

Another nuance is how to initialize values for the target variable in the target sequence, if at all.
- One option is to use teacher forcing during training. Then, at inference, we can initialize the first target value as the last past value, and autoregressively expand the target sequence with model predictions, similar to the LSTM approach above.
  - With this approach, **causal masking** in the decoder is necessary, to ensure attention is only applied to past target timesteps for each target timestep.
  - The **HuggingFace Time Series Transformer** [[7]](#7) seems to implement this approach.
- Another option seems to be simply omitting the target variable representation in the target sequence.
  - iTransformer seems to use this approach, if I understand correctly. So does the **Temporal Fusion Transformer (TFT)** architecture [[8]](#8), though it supports all types of covariates, including future known covariates like the time features in this data.

I wanted a simple method to initialize reasonable target variable values for the target sequence, without needing masking or having to autoregressively collect & use predictions. To this end, I implemented a simple **linear extrapolation** method in my Transformer model.
- Essentially, the target variable values in the source sequence are used to extrapolate "initial" target values for the target sequence.
  - The extrapolation is done by performing linear regression in the matrix form with batched data, using the sequence indices (0 to 72 + 32) as the predictor. It is demonstrated in more detail in the [relevant notebook](https://github.com/AhmetZamanis/DeepLearningEnergyForecasting/blob/main/MISC_TorchLinearExtrapolation.ipynb) in this repository.
- This way, the source sequence consists of past target & covariate values, and the target sequence consists of a future linear trend component & future covariate values.
  - There is no causal masking applied, as the linear extrapolation for all target timesteps are calculated & known at the forecast time.
- Besides the linear extrapolation, one of the future covariates in the model is still a trend dummy that starts from the first timestep in the data. I believe using both together is similar to using a **piecewise trend**, taking into account both the (very) long term trend, and the more short-term, "local" trend.
  - Keep in mind this method may not be appropriate if the source sequence is too short. With 72 timesteps, I believe a linear extrapolation can be reasonably robust, but with much shorter sequences, it may be useless.
  - One could also experiment with using the time series indices, rather than the sequence indices, to perform the linear extrapolation over longer sequences. It could be inefficient to do this within the Torch model though.
- The inspiration for this method was the **Autoformer** architecture [[9]](#9), which essentially employs moving averages in the model architecture to perform trend-cycle decompositions.

Another small modification is a **residual connection** that connects the inverted & projected target sequence (the decoder input) with the decoder output, essentially skipping the attention mechanisms to get to the output layer. This should enable the model to use the simple linear extrapolation as predictions, if a more complex prediction is not warranted.
- The inspiration for this modification is the TFT architecture [[8]](#8), which employs a residual connection that can skip over the entire attention mechanism.

Besides all this, the output layer has the size (timesteps * quantiles), generating multiple quantile predictions for each timestep in the forecast horizon in one go. The loss function & calculation is the same as the LSTM model above.

See below for a diagram of the Linear Inverted Transformer model (best viewed in a new tab).

![Architecture](https://github.com/AhmetZamanis/DeepLearningEnergyForecasting/blob/main/ReportImages/TransformerDiagram.png)

As with the LSTM model, the Transformer hyperparameters were also tuned with Optuna. The best tunes are again on the simpler side:
- One encoder & decoder block often yielded the best performance, usually with eight heads in the multi-attention blocks.
- The attention & feed forward network dimensions were often set to 16 and 8 respectively, downsizing the source & target sequences considerably from 72 and 32.
  - Remember, the sequences are inverted & the timesteps are projected to the attention dimension, unlike the default Transformer.
- A small amount of dropout and exponential learning rate scheduling were again used. The best tunes often trained for around 30-40 epochs, considerably longer than the best LSTM.
- These parameter combinations (and the predictions below) suggest that the Transformer is able to learn more out of the sequences compared to the LSTM, and the attention mechanism does the heavy lifting.

### Performance comparison
For both models, the data was split into source & target sequences of 72 and 32 hours respectively. 
- Each source sequence ends with 16:00, and each target sequence represents the next 8 + 24 hours of comsumption values to be predicted.
- Then, the pairs of source & target sequences (2186 in total) were split into training, validation & testing sets of roughly 60%, 20% and 20% respectively.
- Both models were tuned with the train - validation split. Then, performance testing was performed on the testing set, with the best model hyperparameters, by training on the recombined train & validation sets.
- See notebooks 3.1 and 3.2 for more details on the data handling steps performed before & after testing. Keep in mind the results, plots & metrics in the notebooks may be slightly different from this report, as I experiment further & perform more hyperparameter tuning.

Let's start by comparing the predicted vs. actual values plots for both models over the entire testing set.

<img src="https://github.com/AhmetZamanis/DeepLearningEnergyForecasting/blob/main/ReportImages/LSTMpreds.png" width="500"/> <img src="https://github.com/AhmetZamanis/DeepLearningEnergyForecasting/blob/main/ReportImages/TrafoPreds.png" width="500"/> 

This is a crowded plot, as we have a long, hourly time series. But it still shows how well the model predictions were able to fit the actual values overall.
- Keep in mind the testing set is also split into source & target sequences. For each pair, the models output a prediction only for the target sequence. Hence the gap between actual & predicted values at the start, which is more clearly understood in the plots below.
- From the overall plots, we see the Transformer model generally outputs a wider forecast interval that better contains the actual values. In contrast, the lower bounds of the LSTM forecast interval are often too high.
- Also, the LSTM model is unable to fully capture the spike observed during summer, while the Transformer does so very well.

\
Next, let's zoom into a few source & target sequence pairs along the testing data, compared with the predictions. Of course, we can't do this manually for every pair.

<img src="https://github.com/AhmetZamanis/DeepLearningEnergyForecasting/blob/main/ReportImages/LSTMpreds1.png" width="500"/> <img src="https://github.com/AhmetZamanis/DeepLearningEnergyForecasting/blob/main/ReportImages/TrafoPreds1.png" width="500"/>
<img src="https://github.com/AhmetZamanis/DeepLearningEnergyForecasting/blob/main/ReportImages/LSTMpreds2.png" width="500"/> <img src="https://github.com/AhmetZamanis/DeepLearningEnergyForecasting/blob/main/ReportImages/TrafoPreds2.png" width="500"/> 

We see both models generally do a decent job of predicting the process mean.
- The LSTM predictions are smoother compared to the Transformer predictions, which often capture even the hourly fluctuations very well. Although in some cases, the fluctuations in the Transformer predictions do not seem meaningful.
- Again, the Transformer prediction intervals are wider & do a better job overall of containing the actual values.

\
Finally, let's look at some performance metrics, for both the point & quantile forecasts.
<img src="https://github.com/AhmetZamanis/DeepLearningEnergyForecasting/blob/main/ReportImages/LSTMmetrics.png" width="500"/> <img src="https://github.com/AhmetZamanis/DeepLearningEnergyForecasting/blob/main/ReportImages/TrafoMetrics.png" width="500"/> 

Again, both models perform well, but the Transformer performs considerably better, beating the LSTM in all metrics, point or quantile. Keep in mind that pinball loss at the 50th quantile is essentially half of the MAE.
- Besides the predictive performance, I'd say the Transformer was better in any comparison I can think of during this experiment: It was computationally much more efficient. The model logic, and the data handling & sequencing steps were all more straightforward compared to the LSTM.
- Of course, the best method depends on the problem & data at hand. The LSTM & recurrent architectures still offer the ability to process the data in a strictly sequential manner. For example, this ability is leveraged in the TFT architecture to learn short-term temporal dependencies, while long-term ones are left to the attention mechanism.

### Sources & acknowledgements
<a id="1">[1]<a/> The data was sourced by myself, from the **EPİAŞ Transparency Platform**, which provides open-access data on Türkiye's energy market. The website & API are available in English, though access to the API requires an access application to be made using a static IP address. [Website link](https://seffaflik.epias.com.tr/home)

I have also made the reformatted data available on Kaggle. [Kaggle dataset link](https://www.kaggle.com/datasets/ahmetzamanis/energy-consumption-and-pricing-trkiye-2018-2023/data)
\
\
<a id="2">[2]<a/> A useful article with visualizations that helped me better understand how the stateful LSTM propagates states across batches. [Article link](https://towardsai.net/p/l/stateless-vs-stateful-lstms)
\
\
<a id="3">[3]<a/> D. Salinas, V. Flunkert, J. Gasthaus, DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks, (2019) [arXiv:1704.04110](https://arxiv.org/abs/1704.04110)
\
\
<a id="4">[4]<a/> A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin, Attention Is All You Need, (2023) [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
\
\
<a id="5">[5]<a/> Another excellent article that describes & illustrates the self-attention & cross-attention mechanism very well. [Article link](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)
\
\
<a id="6">[6]<a/> Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, M. Long, iTransformer: Inverted Transformers Are Effective for Time Series Forecasting, (2024) [arXiv:2310.06625](https://arxiv.org/abs/2310.06625)
\
\
<a id="7">[7]<a/> See the HuggingFace documentation for details on the Time Series Transformer, implemented by [Kashif Rasul](https://huggingface.co/kashif). [Documentation link](https://huggingface.co/docs/transformers/model_doc/time_series_transformer)
\
\
<a id="8">[8]<a/> B. Lim, S. O. Arık, N. Loeff, T. Pfister, Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting, (2019) [arXiv:1912.09363](https://arxiv.org/abs/1912.09363)
\
\
<a id="9">[9]<a/> H. Wu, J. Xu, J. Wang, M. Long, Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting, (2022) [arXiv:2106.13008](https://arxiv.org/abs/2106.13008)
\
\
**Dive into Deep Learning** ([d2l.ai](https://d2l.ai/)) is a very comprehensive book that has been my main source for many aspects of deep learning. The book explains recurrent & transformer architectures using language modeling examples.
\
\
[Darts by unit8co](https://unit8co.github.io/darts/index.html) is my go-to Python package for time series forecasting & anomaly detection tasks. It offers Torch & Lightning implementations of many deep learning forecasting models, including a vanilla Transformer & the TFT architecture, both of which I referenced extensively while implementing my own.
\
\
[PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/en/stable/) is a package that offers deep learning time series forecasting capabilities using Torch models. I referenced their TFT & quantile loss implementations during this experiment.

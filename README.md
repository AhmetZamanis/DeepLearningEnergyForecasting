# Introduction

This repository contains the code and results of a multi-step time series forecasting exercise I performed with deep learning models, on a large dataset of hourly energy consumption values.

I used `torch` and `lightning` to implement a stateful LSTM model, and an inverted Transformer model, with some modifications inspired by multiple other time series forecasting architectures. Most notably, I implemented a simple linear extrapolation method in the Transformer model, as a simple way to initialize target variable values for the decoder target sequence.

I also used `Docker` to containerize and deploy the Transformer model. The resulting Docker image can be used to run deployment scripts that automatically update the data from a public API, tune and train a Transformer model, and perform batch predictions. The usage instructions are below, and implementation details are explained in the deployment scripts' & configuration files' comments.

See `Report.md` for an explanation of the models and analysis, along with sources and acknowledgements. See `notebooks/analysis` for the data prep, EDA and modeling notebooks, and `src` for the source code, including Torch classes.

- I also used this dataset and the `GPyTorch` package to try out Gaussian Process Regression with various training strategies. See `notebooks/analysis` for the notebook.
- The merged dataset used in the analysis is available on [Kaggle](https://www.kaggle.com/datasets/ahmetzamanis/energy-consumption-and-pricing-trkiye-2018-2023).

## Instructions: Model deployment with Docker

The simplest way to perform deployment locally is by using the `Make` recipes in the `Makefile`. Run them from the project root.

- Run the Docker recipes to build the service and to run a container. Docker must be installed, and the Docker engine must be running.
- Once a container is running, it stays running unless stopped. Run the deployment script recipes to run the scripts.  
- The inputs, outputs and requirements for every script are in the Makefile comments.
- If Make is not installed, you can copy-paste and run the commands manually, substituting in the service name.

To pull data from the [EPİAŞ API](https://seffaflik.epias.com.tr/home), you need to create an account and add your credentials to a `.env` file in the project root. The format is in `.env-mock`.

The configurations for the deployment scripts are handled with [Hydra](https://hydra.cc/docs/intro/). The config file is in `scripts/deployment/configs/config.yaml`.

- You can update config.yaml in the project root directory, and the configs in the container will also be updated: The two directories are bind mounted by default.
- You can also run the script commands manually without Make, and override configs with Hydra syntax.

You can retain, access and modify the model predictions, datasets, tuning logs and saved models outside the container, in the project root directories. The project root and container directories are bind mounted and have the same structure.

The image is created from an official `CUDA` image. A GPU version of Torch is installed by default. The final image size is around 9GBs. Building the image takes around 7mins on my PC.

- You can use your NVIDIA GPU for training and prediction by default.
  - You may need to modify the relevant fields in `compose.yml` according to the hardware you have.
  - You may need to install additional NVIDIA CUDA tools, depending on your operating system.
    - On Windows, it should be enough to make sure the NVIDIA drivers are installed, and Docker is running with WSL 2. See [the Docker documentation](https://docs.docker.com/desktop/gpu/) for details.
- You can also modify or override the configs to use your CPU only. In that case, I also suggest modifying the base image and Torch installation in the `Dockerfile`, to avoid an unnecessarily large image.

## Instructions: Editable install for development

To modify the code and run the analysis notebooks, you can install the project directly without Docker. I suggest using an editable install, so any changes to the code in `src` take effect immediately.

- Install Python on your system. I used version `3.12` for the analysis, but it's not a strict requirement.
- If you want to use an NVIDIA GPU, install CUDA on your system.
- Create and activate a virtual environment in the project root.
- Perform editable install in project root with `pip install --editable .`
  - To install optional dependencies, use `pip install --editable .[analysis]`. These are only necessary for the GP and Darts notebooks.
- Install a Torch version according to your training device and CUDA version: See [official Torch instructions](https://pytorch.org/get-started/locally/).

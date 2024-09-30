SERVICE_NAME=energy-forecaster

# DOCKER COMMANDS

# Build / rebuild a Docker service from the image
build_service:
	docker compose build

# Build / rebuild service, create, start & attach a container
# Containers stop after "up" is complete, unless overridden by "command", "CMD", "ENTRYPOINT" etc in Dockerfile / compose file.
# Existing containers are reused. If needed, recreated
run_container:
	docker compose up

# Stop a running Docker container
stop_container:
	docker compose stop

# Remove a stopped Docker container
remove_container:
	docker compose rm -f -s

# Stop & remove Docker containers, networks, image associated with the service
remove_service:
	docker compose down --rmi all

# Check if NVIDIA GPU is available in container
check_nvidia:
	docker compose exec $(SERVICE_NAME) nvidia-smi

# docker compose exec reuses an existing container, but doesn't rerun it, it has to be running beforehand.
# docker compose run creates & runs a new container for each execution.


# DEPLOYMENT SCRIPT COMMANDS

# Update raw consumption data
# Requires: EPİAŞ credentials in .env
# Creates: data/deployment/raw/consumption.csv
update_raw_data:
	docker compose exec $(SERVICE_NAME) python3 scripts/deployment/update_raw_data.py 


# Update training data
# Requires: data/deployment/raw/consumption.csv
# Creates: data/deployment/processed/training_data.csv
update_training_data:
	docker compose exec $(SERVICE_NAME) python3 scripts/deployment/update_training_data.py


# Perform model tuning with Optuna, save tuning log
# Requires: data/deployment/processed/training_data.csv
# Creates: data/deployment/tuning-logs/transformer_YYYY-mm-dd_HH-MM-SS.csv
tune_model:
	docker compose exec $(SERVICE_NAME) python3 scripts/deployment/tune_model.py


# Train final model with best tune from last log, save model & scaler
# Requires: 
#	data/deployment/processed/training_data.csv
#	data/deployment/tuning-logs/transformer_YYYY-mm-dd_HH-MM-SS.csv
# Creates:
#	models/deployment/transformer/YYYY-mm-dd_HH-MM-SS.ckpt
# 	models/deployment/scaler/YYYY-mm-dd_HH-MM-SS.joblib
train_model:
	docker compose exec $(SERVICE_NAME) python3 scripts/deployment/train_model.py


# Predict the next H timesteps after the training data, using last saved model & scaler
# Requires:
#	data/deployment/processed/training_data.csv
#	models/deployment/transformer/YYYY-mm-dd_HH-MM-SS.ckpt
# 	models/deployment/scaler/YYYY-mm-dd_HH-MM-SS.joblib
# Creates:
#	data/deployment/predictions/YYYY-mm-dd_HH-MM-SS.csv
batch_predict:
	docker compose exec $(SERVICE_NAME) python3 scripts/deployment/batch_predict.py
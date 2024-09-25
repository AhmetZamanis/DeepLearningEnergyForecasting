SERVICE_NAME=energy-forecaster

# Build / rebuild a Docker service from the image
build_image:
	docker compose build

# Build, create, start & attach a container for the service
# Containers stop after command is complete, unless overridden by "command", "CMD", "ENTRYPOINT" etc.
# Existing containers are reused. If needed, recreated
run_container:
	docker compose up

# Stop a running Docker container
stop_container:
	docker compose stop

# Remove a stopped Docker container
remove_container:
	docker compose rm -f -s

# Stop & remove Docker containers, networks, image associated with the sevice
remove_image:
	docker compose down --rmi all

# docker compose exec reuses an existing container, but doesn't rerun it.
# docker compose run creates a new one each time.

# Update raw consumption data
update_raw_data:
	docker compose exec $(SERVICE_NAME) python scripts/deployment/update_raw_data.py 

# Update training data
update_training_data:
	docker compose exec $(SERVICE_NAME) python scripts/deployment/update_training_data.py
NAME=aesthetic-predictor-v2_5
TAG=0.1
PARENT_DIRECTORY = $(shell pwd)

.PHONY: build run

build:
	@echo "Building Docker image..."
	cp -r ./src ./docker && cp ./pyproject.toml ./docker && cp ./README.md ./docker; \
	cd docker && docker build -t ${NAME}:${TAG} -f Dockerfile .; \
	rm -rf ./src && rm ./pyproject.toml && rm ./README.md
	@echo "Docker image built"

run:
	@if [ `docker container ls -a --filter "name=${NAME}" | wc -l | sed "s/ //g"` -eq 2 ]; then \
		docker container stop ${NAME}; \
		docker container rm ${NAME}; \
		docker container run \
			-it --privileged \
			--gpus all \
			--shm-size=2g \
			-d \
			-v ${PARENT_DIRECTORY}:/app \
			-v /var/run/docker.sock:/var/run/docker.sock \
			--name ${NAME} ${NAME}:${TAG} \
			/bin/bash; \
	else \
		docker container run \
			-it --privileged \
			--gpus all \
			--shm-size=2g \
			-d \
			-v ${PARENT_DIRECTORY}:/app \
			-v /var/run/docker.sock:/var/run/docker.sock \
			--name ${NAME} ${NAME}:${TAG} \
			/bin/bash; \
	fi

log:
	docker container logs -f ${NAME}

up-dist:
	pip install hatch twine
	hatch build
	twine upload dist/*

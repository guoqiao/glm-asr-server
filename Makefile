SHELL = bash

# load .env file and export all vars
include .env
export

app := glm-asr-server
# pass --gpus=all only when nvidia-smi available
gpus := $(shell which nvidia-smi > /dev/null && echo "--gpus=all" || echo "")

# if Linux, which normally means the connection is on a remote vm, no browser
launch_browser := $(shell [ "${os}" = "Linux" ] && echo --no-launch-browser)

platform := --platform=linux/amd64

docker_run := docker run -it --rm \
				-v ~/.cache/huggingface:/root/.cache/huggingface \
				-v .:/workspace \
				--env-file .env \
				--shm-size=2g \
				${gpus} \
				${platform}

.PHONY: all clean test build shell gradio ckpts config

chown:
	sudo chown -R ${USER}:${USER} .

init: chown
	touch .env

prune:
	docker image prune -f

latest:
	docker images --filter "reference=${app}:latest"

# build with no cache:
# export NO_CACHE=--no-cache
build: init prune
	time docker build ${NO_CACHE} \
		${platform} \
		--progress=plain \
		-t ${app}:latest \
		.
	make latest

build-no-cache:
	export NO_CACHE=--no-cache; make build

show-images:
	docker images | grep ^${app}

shell: init
	${docker_run} \
		--entrypoint=/bin/bash \
		${app}:latest

PORT ?= 8000

server:
	${docker_run} \
		-e PORT=${PORT} \
		-p ${PORT}:${PORT} \
		--entrypoint=/opt/conda/bin/python \
		--name=${app} \
		${app}:latest \
		server.py

run: server


tail:
	docker logs -f ${app}


test:
	curl -s -X POST "http://localhost:8000/v1/audio/transcriptions" \
	-H "Content-Type: multipart/form-data" \
	-F "file=@data/audio.mp3" | jq

pre-commit-setup:
	pip install -U pre-commit ruff
	pre-commit install

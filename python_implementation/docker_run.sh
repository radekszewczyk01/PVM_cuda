#!/bin/bash

xhost +
docker run -it --gpus all -w /pvm \
	-v "$(pwd)":/pvm \
	-v "$(realpath "$(pwd)/../PVM_data")":/pvm_data:ro \
        -e DISPLAY=$DISPLAY \
        -e XAUTHORITY=$XAUTH \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v $XAUTH:$XAUTH \
	pvm_docker /bin/bash

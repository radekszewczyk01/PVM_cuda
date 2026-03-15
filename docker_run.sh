#!/bin/bash

xhost +
docker run -it --gpus '"device=1"' -w /pvm \
	-v .:/pvm  \
        -e DISPLAY=$DISPLAY \
        -e XAUTHORITY=$XAUTH \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v $XAUTH:$XAUTH \
	pvm_docker /bin/bash

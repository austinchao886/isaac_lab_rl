#!/bin/bash
export DISPLAY=:1
xhost +local:
docker run -dit --rm \
    --entrypoint bash \
    --runtime=nvidia --gpus all \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -e "DISPLAY=$DISPLAY" \
    -e OMNI_KIT_ALLOW_ROOT=1 \
    -e "ACCEPT_EULA=Y" \
    --network=host \
    -e "PRIVACY_CONSENT=Y" \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    -v /home/austin/Documents/rl_dog_ws/scripts:/scripts \
    -w /scripts \
    --name isaac-lab \
    isaac_lab:dev
    # nvcr.io/nvidia/isaac-sim:4.5.0
    # isaac_lab:dev_atomics
    # isaac_lab:dev_lcm

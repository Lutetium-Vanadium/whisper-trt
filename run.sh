#! /usr/bin/env bash

jetson-containers run \
    -v ~/cache:/model-cache \
    wtrt:r36.4.3 $@
    # --device=/dev/ttyUSB0 \

    #-v /dev/ttyUSB0:/dev/ttyUSB0 \

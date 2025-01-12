#!/bin/bash

docker run -it \
    -v $PWD/fcode:/root/ \
    ftorch_online /bin/bash


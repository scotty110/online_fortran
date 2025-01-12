#!/bin/bash
rm -rf build

set -eu

mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=./ -DCMAKE_BUILD_TYPE=Release
cmake --build .
#!/usr/bin/env sh
set -e

./build/tools/caffe train \
  --solver=SF/mnist/lenet_consolidated_solver.prototxt $@

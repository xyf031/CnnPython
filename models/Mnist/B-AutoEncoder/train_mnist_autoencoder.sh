#!/usr/bin/env sh
set -e

nohup ./build/tools/caffe train \
  --solver=SF/mnist/mnist_autoencoder_solver.prototxt $@ >& SF/mnist/log-mnist_autoencoder_solver.txt &

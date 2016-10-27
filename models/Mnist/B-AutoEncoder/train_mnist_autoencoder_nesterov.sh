#!/bin/bash
set -e

nohup ./build/tools/caffe train \
  --solver=SF/mnist/mnist_autoencoder_solver_nesterov.prototxt $@ >& SF/mnist/mnist_autoencoder_solver_nesterov.txt &

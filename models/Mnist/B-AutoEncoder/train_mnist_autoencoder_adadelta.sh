#!/bin/bash
set -e

nohup ./build/tools/caffe train \
  --solver=SF/mnist/mnist_autoencoder_solver_adadelta.prototxt $@ >& SF/mnist/log-mnist_autoencoder_solver_adadelta.txt &

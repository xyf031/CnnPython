#!/bin/bash
set -e

nohup ./build/tools/caffe train \
  --solver=SF/mnist/mnist_autoencoder_solver_adagrad.prototxt $@ >& SF/mnist/log-mnist_autoencoder_solver_adagrad.txt &

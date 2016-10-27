#!/usr/bin/env sh
set -e

nohup ./build/tools/caffe train --solver=SF/mnist/lenet_solver_adam.prototxt $@ >& SF/mnist/log-lenet_solver_adam.txt &

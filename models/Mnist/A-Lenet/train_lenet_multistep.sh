#!/usr/bin/env sh
set -e

nohup ./build/tools/caffe train --solver=SF/mnist/lenet_multistep_solver.prototxt $@ >& SF/mnist/log-lenet_multistep_solver.txt &

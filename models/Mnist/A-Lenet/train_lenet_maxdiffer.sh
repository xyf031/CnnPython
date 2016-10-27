#!/usr/bin/env sh
set -e

nohup ./build/tools/caffe train --solver=SF/mnist/lenet_solver_maxdiffer.prototxt $@ >& SF/mnist/log-lenet_solver_maxdiffer.txt &

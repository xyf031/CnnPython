#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=SF/lenetSF-solver.prototxt -gpu 0


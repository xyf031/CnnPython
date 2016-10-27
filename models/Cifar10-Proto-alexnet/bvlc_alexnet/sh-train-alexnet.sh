
set -e

TOOLS=./build/tools

nohup $TOOLS/caffe train --solver=SF/cifar10-alexnet/solver-alexnet-cifar10.prototxt >& SF/cifar10-alexnet/log-solver-alexnet-cifar10.txt &

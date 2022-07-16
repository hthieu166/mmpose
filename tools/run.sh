#!/bin/bash
NUM_GPUS=1
CONFIG_FILE=$2

case $1 in

    "train")
        echo "Run training pipeline"
        TRAIN=1
        TEST=0
        ;;

    "test")
        echo "Run testing pipeline"
        TRAIN=0
        TEST=1
        ;;

    "all")
        echo "Run training and testing pipeline"
        TRAIN=1
        TEST=1
        ;;
esac

# set CUDA_VISIBLE_DEVICES 
if [ -z $3 ]
then
    CUDA_VISIBLE_DEVICES=0
else
    CUDA_VISIBLE_DEVICES=$3
fi
# set PORT
if [ -z $4 ]
then
    PORT=${PORT:-29500}
else
    PORT=$4
    echo "Using PORT: $PORT"
fi

if [ $TRAIN = 1 ]
then
    ## training
    MKL_SERVICE_FORCE_INTEL=1 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES\
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT \
        $(dirname "$0")/train.py $CONFIG_FILE --launcher pytorch
fi

if [ $TEST  = 1 ]
then
    ## testing
    MKL_SERVICE_FORCE_INTEL=1 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES\
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT \
        $(dirname "$0")/test.py $CONFIG_FILE --launcher pytorch \
        # -C $CHECKPOINT
fi
#!/bin/bash

OUTPUT_DIR=data/unet_muscels2
DATA_DIR=datasets/left_2/imagesTs

CKP_DIR=$OUTPUT_DIR/ckp
LOG_DIR=$OUTPUT_DIR/logs
OUT_DIR=$OUTPUT_DIR/output

RESIZE=128
DEVICE="cpu"

echo "Creating directories in $OUTPUT_DIR"
mkdir -p $OUT_DIR
mkdir -p $LOG_DIR

declare -a default_args=("--debug" "infer" "--resize=$RESIZE" "--data-dir=$DATA_DIR" "--device=$DEVICE" "--metrics")

echo "Loading models from $CKP_DIR"

for x in $CKP_DIR/*.pth; do
    x=${x%.pth}
    x=${x##*/}
    echo "================== Running on $x =================="
    args=("${default_args[@]}")

    NETWORK="unet"

    function get_network() {
        if [[ $x == *"deepatlas"* ]]; then
            echo "deepatlas"
        elif [[ $x == *"unetr"* ]]; then
            echo "unetr"
        elif [[ $x == *"uneta"* ]]; then
            echo "uneta"
        else
            echo "unet"
        fi
    }
    # set NETWORK TO get_network
    NETWORK=$(get_network)

    if [[ $NETWORK == "deepatlas" ]]; then
        echo "Loading $x with deepatlas"
    else
        echo "Loading $x with $NETWORK"
        args=("${args[@]}" "--network=$NETWORK" "--solo-seg")
    fi

    if [[ $x == *"2ch"* ]]; then
        echo "Loading $x with 2 channels"
        args=("${args[@]}" "--add-cm-ch")
    else
        echo "Loading $x with 1 channel"
    fi

    # if [[ $x == *"conf"* ]]; then
    #     echo "Loading $x with confidence maps in loss"
    #     args=("${args[@]}" "--add-cm-loss")
    # else
    #     echo "Loading $x without confidence maps in loss"
    # fi

    echo "Running : python -m deepatlas.main ${args[@]} --seg-model $CKP_DIR/$x.pth >$LOG_DIR/$x.log"

    python -m deepatlas.main "${args[@]}" --seg-model $CKP_DIR/$x.pth >$LOG_DIR/$x.log

    if [ $? -eq 0 ]; then
        mkdir -p $OUT_DIR/$x && echo "Created $OUT_DIR/$x"

        mv $DATA_DIR/labels/pred/*.gz $OUT_DIR/$x/ && echo "Moved to $OUT_DIR/$x"
    else
        echo FAIL
    fi
done

echo "Running results.py to generate csv and boxplots"
python scripts/results.py $LOG_DIR

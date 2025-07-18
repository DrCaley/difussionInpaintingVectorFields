#!/usr/bin/env bash

PKG_ROOT="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/.."

cd $PKG_ROOT
EXPERIMENT_NAME="divergence_weight"

for CFG_FILE in $(ls $PKG_ROOT/cfg/$EXPERIMENT_NAME); do
    CFG_PATH=$(pwd)/cfg/$EXPERIMENT_NAME/$CFG_FILE

    echo $CFG_PATH

    python3 ddpm/training/xl_ocean_trainer.py \
        --model_name ${EXPERIMENT_NAME}_${CFG_FILE%.*} \
        --training_cfg $CFG_PATH
done
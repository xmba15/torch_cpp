#!/usr/bin/env bash

git submodule deinit -f .
git submodule update --init --recursive

readonly CURRENT_DIR=$(dirname $(realpath $0))
readonly ROOT_DIR=$(realpath "$CURRENT_DIR"/..)

cd $ROOT_DIR/scripts/superglue/SuperGluePretrainedNetwork
echo "apply patch" && git apply ../jit_patch.patch
python3 -m pip install -r $ROOT_DIR/scripts/superglue/SuperGluePretrainedNetwork/requirements.txt --user

cd $ROOT_DIR/data
python3 $ROOT_DIR/scripts/superglue/jit_superglue_model.py
python3 $ROOT_DIR/scripts/superglue/jit_superpoint_model.py

#!/usr/bin/env bash

readonly PANGOLIN_URL="https://github.com/stevenlovegrove/Pangolin.git"
readonly PANGOLIN_VERSION="v0.8"

cd /tmp
git clone --recursive -b $PANGOLIN_VERSION $PANGOLIN_URL
cd Pangolin
./scripts/install_prerequisites.sh --dry-run recommended
mkdir build && cd build
cmake ../
make -j`nproc`
sudo make install
sudo ldconfig

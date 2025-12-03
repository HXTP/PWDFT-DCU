#! /bin/bash

source ./env.sh
echo ${DGDFT_DIR}

#rm -rf  ${DGDFT_DIR}/external/yaml-cpp
#cd ${DGDFT_DIR}/external/yaml-cpp-0.7.0
#rm -rf build/*
#cd build
#cmake -DCMAKE_INSTALL_PREFIX=${DGDFT_DIR}/external/yaml-cpp ..
#make
#make install

#cd  ${DGDFT_DIR}/external/lbfgs
#make cleanall && make
#cd  ${DGDFT_DIR}/external/rqrcp
#make cleanall && make
#cd  ${DGDFT_DIR}/external/blopex/blopex_abstract
#make clean && make
cd  ${DGDFT_DIR}/src
 make -j
cd  ${DGDFT_DIR}/examples
 make pwdft  #&& make dgdft

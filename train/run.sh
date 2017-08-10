#!/bin/bash

./code/get_list.py /home/bjohnson/software/facenet/data/casia_maxpy_mtcnnpy_112 > data/casia.txt

./code/sphereface/sphereface_train.sh 0

# 
./../tools/caffe-sphereface/build/tools/caffe train \
    -solver ./code/sphereface/sphereface_solver.prototxt.2 \
    -snapshot ./result/sphereface/sphereface_model_iter_28000.solverstate \
    -gpu 0 2>&1 | tee result/sphereface/sphereface.log.2

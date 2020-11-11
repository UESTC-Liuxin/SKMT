#!/usr/bin/env bash

GPUS="0"
export CUDA_VISIBLE_DEVICES=$GPUS
python main.py --auxiliary "fcn" \
              --batch_size 2   \
              --crop_size 512 \
              --image_size 512 \
              --max_epochs 200 \
              --num_classes 19 \
              --lr 0.004  \
              --savedir "./runs/deeplab_resnet/" \
              --gpus $GPUS
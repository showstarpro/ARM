#!/bin/bash
# bash /client-tools/repair_A100.sh
# source /root/anaconda3/bin/activate /root/anaconda3/envs/mam
# cd /lpai/ARM/det;

DET_CONFIG_NAME=cascade_mask_rcnn_vimdet_b_100ep
DET_CONFIG=projects/ViTDet/configs/COCO/$DET_CONFIG_NAME.py


python tools/lazyconfig_train_net.py \
 --num-gpus 8 --num-machines 1 --machine-rank 0 --dist-url "tcp://127.13.44.12:60900" \
 --config-file ${DET_CONFIG} \
 train.output_dir=/lpai/ARM/det/work_dirs/$DET_CONFIG_NAME-4gpu-test-env \
 dataloader.train.num_workers=128 \
 dataloader.test.num_workers=8

 
# /root/anaconda3/envs/mam/bin/python tools/lazyconfig_train_net.py \
#  --num-gpus 1 --num-machines 1 --machine-rank 0 --dist-url "tcp://127.13.44.12:60900" \
#  --config-file ${DET_CONFIG} \
#  train.output_dir=/lpai/ARM/det/work_dirs/$DET_CONFIG_NAME-4gpu-test-env \
#  dataloader.train.num_workers=128 \
#  dataloader.test.num_workers=8

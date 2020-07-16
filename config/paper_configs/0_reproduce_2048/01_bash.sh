#!/bin/bash
set -ex

gpu="4"

CUDA_VISIBLE_DEVICES=${gpu} python run_nerf.py --config ./config/paper_configs/0_reproduce_2048/config0_rs_fern.txt \
--no_ndc --spherify --lindisp

CUDA_VISIBLE_DEVICES=${gpu} python run_nerf.py --config ./config/paper_configs/0_reproduce_2048/config1_rs_flower.txt \
--no_ndc --spherify --lindisp

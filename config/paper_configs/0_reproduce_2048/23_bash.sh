#!/bin/bash
set -ex

gpu="5"

CUDA_VISIBLE_DEVICES=${gpu} python run_nerf.py --config ./config/paper_configs/0_reproduce_2048/config2_rs_fortress.txt \
--no_ndc --spherify --lindisp

CUDA_VISIBLE_DEVICES=${gpu} python run_nerf.py --config ./config/paper_configs/0_reproduce_2048/config3_rs_horn.txt \
--no_ndc --spherify --lindisp

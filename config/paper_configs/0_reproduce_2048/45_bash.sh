#!/bin/bash
set -ex

gpu="6"

CUDA_VISIBLE_DEVICES=${gpu} python run_nerf.py --config ./config/paper_configs/0_reproduce_2048/config4_rs_leaves.txt \
--no_ndc --spherify --lindisp

CUDA_VISIBLE_DEVICES=${gpu} python run_nerf.py --config ./config/paper_configs/0_reproduce_2048/config5_rs_orchids.txt \
--no_ndc --spherify --lindisp

#!/bin/bash
set -ex

gpu="7"

CUDA_VISIBLE_DEVICES=${gpu} python run_nerf.py --config ./config/paper_configs/reproduce_2048/config6_rs_room.txt \
--no_ndc --spherify --lindisp

CUDA_VISIBLE_DEVICES=${gpu} python run_nerf.py --config ./config/paper_configs/reproduce_2048/config7_rs_trex.txt \
--no_ndc --spherify --lindisp

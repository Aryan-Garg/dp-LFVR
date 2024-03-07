#!/usr/bin/env python3

import sys
import os
import subprocess
import glob
from tqdm.auto import tqdm 

def getAblationModels(vers, gpu):
    if vers == "v1":
        os.system(f"./train.py -e 30 --validate_every 1 -exp cont_{vers}_labl -r -rid v1_labl@27-Oct_12:30:44 \
              --gpu {gpu} --filenames_file_folder ./train_inputs/Pixel4_3DP_frame_skip10 \
              -bs 1 \
              --loss_ablation {vers}")
    elif vers == "v4":
        os.system(f"./train.py -e 30 --validate_every 1 -exp cont_{vers}_labl -r -rid v4_labl@27-Oct_12:32:02 \
              --gpu {gpu} --filenames_file_folder ./train_inputs/Pixel4_3DP_frame_skip10 \
              -bs 1 \
              --loss_ablation {vers}")
    elif vers == "v2":
        os.system(f"./train.py -e 30 --validate_every 1 -exp cont_{vers}_labl -r -rid v2_labl@30-Oct_10:21:59 \
                --gpu {gpu} --filenames_file_folder ./train_inputs/Pixel4_3DP_frame_skip10 \
                -bs 1 \
                --loss_ablation {vers}")
    else:
        os.system(f"./train.py -e 30 --validate_every 1 -exp {vers}_labl \
              --gpu {gpu} --filenames_file_folder ./train_inputs/Pixel4_3DP_frame_skip10 \
              -bs 1 \
              --loss_ablation {vers}")

if __name__ == '__main__':
    versions = ["v6"] # ["v2", "v1", "v3"]# "v4", "v5", "v6"]
    for vers in versions:
        getAblationModels(vers, gpu=2)
#!/usr/bin/env python

import os
import sys
import random

# format: tab separated dataset/test/{}
def main(dataPath, ds):
    os.makedirs(f"{ds}_Seq", exist_ok=True) 
    with open(f"{ds}_Seq/test_files.txt", "w") as f:
        all_seqs = os.listdir(f"{dataPath}/{ds}/test")
        for seq in all_seqs:
            all_lfs = sorted(os.listdir(f"{dataPath}/{ds}/test/{seq}"))
            if len(all_lfs) < 3:
                continue
            for lf in all_lfs[:3]:
                lfX = lf.split('.')[0]
                f.write(f"{ds}/test/{seq}/{lf}\t{ds}/test/left_{lfX}.png\t{ds}/test/right_{lfX}.png\t{ds}/test/left_{lfX}_disp.png\n")


    # with open(f"{ds}/train_files.txt", "w") as f:
    #     all_lfs = os.listdir(f"{dataPath}/{ds}/train/")
    #     for lf in all_lfs:
    #         lfX = lf.split('.')[0]
    #         f.write(f"{ds}/train/{lf}\t{ds}/train/left_{lfX}.png\t{ds}/train/right_{lfX}.png\t{ds}/train/left_{lfX}_disp.png\n")
            

if __name__ == '__main__':
    dataPath = "/media/data/prasan/datasets/LF_video_datasets"
    datasets = ["Stanford", "TAMULF", "Hybrid", "Kalantari"]
    for ds in datasets:
        main(dataPath, ds)

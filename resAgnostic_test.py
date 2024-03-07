#!/usr/bin/env python3

import sys
import os
import subprocess
import glob
from tqdm.auto import tqdm 

def getResults(dataset, h, w, sr=False):
    assert dataset is not None, "Please provide dataset name"
    if sr:
        os.system(f"./test.py --calcMetrics -sr \
              -exp AddLN_GA16_RGB+DP_30@21-Aug_17:59:46 \
              -ckpt 093.pt \
              --baseline_factor 1.5 \
              --gpu 2 \
              --dataset {dataset} \
              --genDP_path /data2/aryan/{dataset}/test_dp \
              --filenames_file_eval ./FT_train-test_files/{dataset}/test_files.txt \
              --otherDS_disp_path random_blooah \
              -vh {h} \
              -vw {w}")
    else:
        os.system(f"./test.py --calcMetrics\
              -exp AddLN_GA16_RGB+DP_30@21-Aug_17:59:46 \
              -ckpt 093.pt \
              --baseline_factor 1.5 \
              --gpu 2 \
              --dataset {dataset} \
              --genDP_path /data2/aryan/{dataset}/test_dp \
              --filenames_file_eval ./FT_train-test_files/{dataset}/test_files.txt \
              --otherDS_disp_path /data2/aryan/unimatch/dp_otherDS/ \
              -vh {h} \
              -vw {w}")
    # --disp_input
    # AddLN_GA16_RGB+DP_30@21-Aug_17:59:46
    # AddLN_GA16_RGB+DP_30@18-Aug_15:08:35

def getVideoResults(ds, h, w):
    dataset = ds + "_Seq"
    os.system(f"./train.py --mode test --save_lf --gpu 1 -r --conf_save ours_kalantari \
              -rid AddLN_GA16_RGB+DP+Disp_30@18-Aug_15:08:35 -bs 1 --dataset {dataset} --depth_input -th {h} -tw {w} -vh {h} -vw {w} \
                --filenames_file_folder ./FT_train-test_files/{dataset}/ --baseline 2.5")

# def dispMethodComparison(dataset):
#     assert dataset is not None, "Please provide dataset name"

#     rh, rw = 384, 528 # Lytro Illum Resolution
#     disp_paths = ['/media/data/prasan/datasets/LF_datasets/DPT-depth/', '/data2/aryan/unimatch/dp_otherDS/']

#     for dips_path in disp_paths:
#         os.system(f"./test.py --calcMetrics \
#               --gpu 2 \
#               --dataset {dataset} \
#               --genDP_path /data2/aryan/{dataset}/test_dp \
#               --filenames_file_eval ./FT_train-test_files/{dataset}/test_files.txt \
#               -vh {rh} \
#               -vw {rw} \
#               --otherDS_disp_path {dips_path}")

def calcTimes(dataset, h, w):
    assert dataset is not None, "Please provide dataset name"

    os.system(f"./test.py --calcTime --calcMetrics \
              --gpu 2 \
              --dataset {dataset} \
              --genDP_path /data2/aryan/{dataset}/test_dp \
              --filenames_file_eval ./FT_train-test_files/{dataset}/test_files.txt \
              -vh {h} \
              -vw {w}")

if __name__ == '__main__':
    datasets = ['Kalantari', 'TAMULF', 'Hybrid', 'Stanford']
    # datasets = ['Kalantari']
    # our: 256x192 | Govindrajan: 176x264 | Li: 192x192 | Srinivasan: 188x270 |
    # LIC: 352x528 (to introduce another baseline) | 480p-SD: 480x640

    # resH = [480]
    # resW = [640]

    # resH2 = [256, 176, 192, 352] # 480
    # resW2 = [192, 264, 192, 528] # 640

    # plot_dim = [192, 256, 384, 480] #, 576]
    
    resH = [480]
    resW = [640]
    for dataset in (pbar:=tqdm(datasets)):
        pbar.set_description(f"Processing {dataset}")
        for i in range(len(resH)):
            getResults(dataset, resH[i], resW[i], sr=True)
            # calcTimes(dataset, resH2[i], resW2[i])
            
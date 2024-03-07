#!/usr/bin/env python3

import sys
import os
import subprocess
import glob
from tqdm.auto import tqdm 

def getResults(dataset, h, w, sr=False, exp=None, ckpt=None):
    assert exp is not None, "Please provide experiment name"
    assert ckpt is not None, "Please provide checkpoint name"
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
        print(f"Doing exp: {exp} with {ckpt}")
        os.system(f"./test.py --calcMetrics\
              -exp {exp} \
              -ckpt {ckpt} \
              --baseline_factor 1.5 \
              --results metrics \
              --gpu 2 \
              --dataset {dataset} \
              --genDP_path /data2/aryan/{dataset}/test_dp \
              --filenames_file_eval ./FT_train-test_files/{dataset}/test_files.txt \
              --otherDS_disp_path lol \
              -vh 256 \
              -vw 192")
    # --disp_input
    # AddLN_GA16_RGB+DP_30@21-Aug_17:59:46
    # AddLN_GA16_RGB+DP_30@18-Aug_15:08:35

def getVideoResults(ds, h, w):
    dataset = ds + "_Seq"
    os.system(f"./train.py --mode test --save_lf --gpu 1 -r --conf_save FbyF_temp_results2 \
              -rid AddLN_GA16_RGB+DP_30@21-Aug_17\:59\:46 -bs 1 --dataset {dataset} -th {h} -tw {w} -vh {h} -vw {w} \
                --filenames_file_folder ./FT_train-test_files/{dataset}/")


if __name__ == '__main__':
    datasets = ['Hybrid', 'Stanford', 'Kalantari', 'TAMULF']
    # datasets = ['Kalantari']
    # our: 256x192 | Govindrajan: 176x264 | Li: 192x192 | Srinivasan: 188x270 |
    # LIC: 352x528 (to introduce another baseline) | 480p-SD: 480x640

    # resH = [352, 480]
    # resW = [528, 640]

    # resH2 = [256, 176, 192, 352] # 480
    # resW2 = [192, 264, 192, 528] # 640

    # plot_dim = [192, 256, 384, 480] #, 576]
    
    resH = [256]
    resW = [192]
    exps = ["v1_labl@27-Oct_12:30:44", "v2_labl@30-Oct_10:21:59", "v3_labl@02-Nov_13:35:55", 
                 "v4_labl@27-Oct_12:32:02", "v5_labl@02-Nov_02:38:01", 
                 "v6_labl@06-Nov_10:20:57"
                 ]
    ckpts = ["024.pt", "029.pt","019.pt", "026.pt","024.pt","026.pt"]
    for exp, ckpt in zip(exps, ckpts):
        for dataset in (pbar:=tqdm(datasets)):
            pbar.set_description(f"Processing {dataset}")
            for i in range(len(resH)):
                getResults(dataset, resH[i], resW[i], sr=False, exp=exp, ckpt=ckpt)
            # calcTimes(dataset, resH2[i], resW2[i])
            
#!/usr/bin/env python

import os
import sys
import numpy as np
import argparse
import time
import cv2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss_temp import TemporalConsistency


def getLF(path, isPred=False):
    lf = np.load(path)
    H, W = 256, 192
    new_lf = np.zeros((49, H, W, 3))
    if not isPred:
        u, v, h, w, c = lf.shape
        assert u*v == 49, "LF shape is not 7x7"
        if c != 3:
            u, v, c, h, w = lf.shape
        lf = lf.reshape(u*v, h, w, c)
        # resize lf
        for i in range(49):
            tmp_sai = cv2.resize(lf[i], (W, H), interpolation=cv2.INTER_CUBIC) # w x h
            new_lf[i] = tmp_sai
        new_lf = new_lf.reshape(u*v, c, H, W)
        new_lf = (new_lf - np.amin(new_lf)) / (np.amax(new_lf) - np.amin(new_lf))
    else:
        new_lf = lf
        new_lf = (new_lf - np.amin(new_lf)) / (np.amax(new_lf) - np.amin(new_lf))
    
    new_lf = new_lf.reshape(7, 7, 3, H, W)
    # print(new_lf.shape, np.amax(new_lf), np.amin(new_lf))
    return torch.Tensor(new_lf).to(device)
    


def main(dataset, args, device):
    seqs = os.listdir(PRED_LF_PATH + dataset + "_Seq/")
    temporal_loss_avg = []
    temporal_loss = TemporalConsistency(args, device)

    for seq in (pbar := tqdm(seqs)):
        # check if seq is a directory
        if not os.path.isdir(PRED_LF_PATH + dataset + f"_Seq/{seq}"):
            continue
        pred_lfs = sorted(os.listdir(PRED_LF_PATH + dataset + f"_Seq/{seq}"))
        gt_lfs = sorted(os.listdir(GT_LF_PATH + dataset + f"/test/{seq}"))

        for i in range(1, len(pred_lfs), 2):
            # print(f"Processing {pred_lfs[i]} and {pred_lfs[i-1]}")
            # print(f"GT: Processing {gt_lfs[i]} and {gt_lfs[i-1]}")
            gt_lf = getLF(GT_LF_PATH + dataset + f"/test/{seq}/" + gt_lfs[i])
            prev_lf = getLF(GT_LF_PATH + dataset + f"/test/{seq}/" + gt_lfs[i-1])
            pred_lf = getLF(PRED_LF_PATH + dataset + f"_Seq/{seq}/" + pred_lfs[i], isPred=True)
            # print(gt_lf.shape, prev_lf.shape, pred_lf.shape)

            temp_loss_this = temporal_loss(gt_lf, prev_lf, pred_lf)
            temporal_loss_avg.append(temp_loss_this.item())
        pbar.set_description(f"tl: {np.mean(temporal_loss_avg):.3f}")
    
    with open(f"./FbyF_temp_results2/{dataset}_Seq/temporal_loss.txt", "w") as f:
        f.write(f"Temporal Loss: {np.mean(temporal_loss_avg)}")

    return np.mean(temporal_loss_avg)

if __name__ == '__main__':
    GT_LF_PATH = "/media/data/prasan/datasets/LF_video_datasets/"
    PRED_LF_PATH = "./FbyF_temp_results2/"

    parser = argparse.ArgumentParser()
    parser.add_argument('--angular', type=int, default=7)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--small', action='store_true', default=False)
    parser.add_argument('--mixed_precision', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    datasets = ["Hybrid", "Kalantari", "TAMULF", "Stanford"]

    final_temp_loss = 0
    for dataset in datasets:
        print("Dataset:", dataset)
        temp_loss_dataset = main(dataset, args, device)
        final_temp_loss += temp_loss_dataset

    print("Avg-ed across dataset sequences temporal Loss:", 1. * final_temp_loss/len(datasets))
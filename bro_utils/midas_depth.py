#!/usr/bin/env python3

import os
import argparse
import glob
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import manip as mp

def load_lf(file):
    plf, nlf = mp.read_lf(file, (7,7,480,640,3))
    img = nlf[3, 3, ...]
    # img = np.transpose(img, (2, 0, 1))
    return img #.to(device)


def normalize(img):
    return (img - np.amin(img)) / (np.amax(img) - np.amin(img))

def infer(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    os.makedirs(args.out_path, exist_ok=True)

    with torch.no_grad():
        lfs = glob.glob(args.lfs_path + f"*.npy", recursive=True)
        print(f"Found {len(lfs)} images. Saving files to {args.out_path}/")

        for file in tqdm(lfs):
            img = load_lf(file)
            # print(img.shape)
            input_batch = transform(img).to(device)
            depth = midas(input_batch)
            depth = torch.nn.functional.interpolate(depth.unsqueeze(1),
                                                    size=(args.height, args.width),
                                                    mode="bicubic", 
                                                    align_corners=False).squeeze()
            output = depth.cpu().numpy() / 30
            # print(output.shape, np.amax(output), np.amin(output))
            # plt.imshow(output * 255)
            # plt.show()
            cv2.imwrite(args.out_path + file.split("_")[-1].split(".")[0] + ".png", output * 255)
            

class ARGS():
    out_path = "./depth_from_central/"
    height = 480
    width = 640
    dataset = "Pixel4"
    lfs_path = "./baseline_1.5_SD/"

if __name__ == '__main__':
    args = ARGS()
    infer(args)

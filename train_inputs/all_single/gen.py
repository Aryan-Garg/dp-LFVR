#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import cv2
import random
import glob

BASE = "/data2/raghav/datasets/Pixel4_3DP/rectified/B/Video_data"
unrectified_videoPaths = "/data2/raghav/datasets/Pixel4_3DP/unrectified/B/dp_data"
with open("test_files.txt", "w") as f:
    for vid in os.listdir(BASE):
        count = 0
        imgs = sorted(os.listdir(os.path.join(BASE, vid)), key= lambda x: random.random())
        urect_vid = sorted(glob.glob(os.path.join(unrectified_videoPaths, vid) + "/*"))
        fileCheckName =  urect_vid[0].split("/")[-1]
        if "right" in urect_vid[0] or "left" in urect_vid[0]:
            for img in imgs:
                if img.endswith(".jpg"):
                    img_name = img.split(".")[0]
                    img_number = int(img_name.split("_")[-1])
                    f.write(f"{vid}/{img};{vid}/{img_name}_left.jpg;{vid}/{img_name}_right.jpg;{vid}/{img}\n")
                    count += 1

                if count >= 1:
                    break



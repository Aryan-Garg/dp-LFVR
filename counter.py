#!/usr/bin/env python3

import os
import sys

import glob


BASE = "/data2/raghav/datasets/Pixel4_3DP/rectified/B/Video_data/"
total_count = 0

for dir in os.listdir(BASE):
    path = os.path.join(BASE, dir)
    count_imgs = 0
    for img in glob.glob(os.path.join(path, "*.jpg")):
        count_imgs += 1
    print(f"{dir}: {count_imgs}")
    total_count += count_imgs

print(f"Total: {total_count}")
#!/usr/bin/env python3

import os

import numpy as np

BASE = "/data/prasan/datasets/LF_video_datasets"
datasets = ["TAMULF", "Stanford", "Kalantari", "Hybrid"]

with open("test_files.txt", "w") as f:
    for ds in datasets:
        for seq in os.listdir(os.path.join(BASE, ds, "test")):
            for lf in os.listdir(os.path.join(BASE, ds, "test", seq)):
                lf_path = os.path.join(ds,"test",seq, lf)
                dummy_str = "3407"
                f.write(lf_path + "\t" + dummy_str + "\t" + dummy_str + "\t" + dummy_str + "\n")
                break

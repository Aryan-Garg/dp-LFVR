#!/usr/bin/env python

import os
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 as cv

# list all dirs in current directory
dirs = [d for d in os.listdir('.') if os.path.isdir(d)]

countDict = {}
for dir in (pbar := tqdm(dirs)):
    pbar.set_description(f"Processing {dir}")
    # read all pfm files in current directory
    files = [f for f in os.listdir(dir) if f.endswith('.png')]
    # plot histogram of all files
    
    for file in (pbar2:=tqdm(files)):
        pbar2.set_description(f"Processing {file}")
        # read pfm file
        disp = cv.imread(os.path.join(dir, file), cv.IMREAD_UNCHANGED)
        # count frequency of each disparity value
        unique, counts = np.unique(disp, return_counts=True)
        for u,c in zip(unique, counts):
            if u in countDict:
                countDict[u] += c
            else:
                countDict[u] = c
    bruh = sorted(list(countDict.keys()))
    print(bruh[0], bruh[-1])
    break
    
# # plot histogram from dictionary
# plt.bar(list(countDict.keys()), list(countDict.values()), color='g')
# plt.title("Unimatch Disp. Histogram")
# plt.xlabel("Disparity")
# plt.ylabel("Frequency")
# plt.savefig("unimatch_hist.png")
# plt.show()

#!/usr/bin/env python

import os
import numpy as np

import cv2
import matplotlib.pyplot as plt


import .manip as mp


plf, nlf = mp.read_lf("./baseline_1.5_SD/LF_11.npy", (7,7,480,640,3))

os.makedirs("RMLFs", exist_ok=True)

left = cv2.cvtColor((255*nlf[3,0,:,:,:]).astype('uint8'), cv2.COLOR_BGR2RGB)
right = cv2.cvtColor((255*nlf[3,6,:,:,:]).astype('uint8'), cv2.COLOR_BGR2RGB)
center = cv2.cvtColor((255*nlf[3,3,:,:,:]).astype('uint8'), cv2.COLOR_BGR2RGB)

cv2.imwrite("RMLFs/left.png", left)
cv2.imwrite("RMLFs/right.png", right)
cv2.imwrite("RMLFs/center.png", center)

# for i in range(7):
#     for j in range(7):
#         num = str((i*7) + j).zfill(3)
#         cv2.imwrite(f"RMLFs/input_Cam{num}.png", cv2.cvtColor((255*nlf[i,j,:,:,:]).astype('uint8'), cv2.COLOR_BGR2RGB))
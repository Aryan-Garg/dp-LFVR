#!/usr/bin/env python3

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from plenpy.lightfields import LightField
from moviepy.editor import *
from tqdm.auto import tqdm

from .manip import *


def get_epi(plf, iterAxis, fixAng, fixSpat):
    # iterAxis = 0 --> Horizontal
    # iterAxis = 1 --> Vertical
    return plf.get_epi(iterAxis, fixAng, fixSpat)
    

def custom_epi(nlf, iterAxis, fixAng, fixSpat, start_px, width=60):
    if iterAxis == 0:
        epi = nlf[:, fixAng, start_px:start_px+width, fixSpat].copy()
    elif iterAxis == 1:
        epi = nlf[fixAng, :, fixSpat, start_px:start_px+width].copy()

    return epi


def show_all_3(center_view, epiV, epi):
    fig, axs = plt.subplots(2,2, figsize=(12, 8))
    axs[0,0].imshow(center_view)
    axs[0,0].axis('off')
    axs[0,0].set_title("Center View")

    axs[0,1].imshow(epiV.transpose(1,0,2))
    axs[0,1].set_title("Vertical EPI")
    axs[0,1].axis('off')

    axs[1,0].imshow(epi.transpose(1,0,2))
    axs[1,0].set_title("Horizontal EPI")
    axs[1,0].axis('off')

    axs[1,1].imshow(np.ones((nlf.shape[2], nlf.shape[3], 3)) * 255)
    axs[1,1].axis('off')
    fig.tight_layout()
    plt.show()

# (u, v, s, t, ch) :: (7,7,480,640,3)

## v -->
## u    t -->
## |    s
## |    |
## |    |
## v    v
# 480x640x3
if __name__ == '__main__':
    plf, nlf = read_lf("./baseline_1.5_SD/LF_11.npy", (7,7,480,640,3))
    start_W, widthW, fixSpat_t = 100, 100, 600 # vertical epi
    start_H, widthH, fixSpat_s = 200, 200, 300 # horizontal epi
# plt.imshow(nlf[3,3,:,:,:])
# plt.show()
    epiV = custom_epi(nlf, 0, 3, fixSpat_t, start_W, widthW)
    epiH = custom_epi(nlf, 1, 3, fixSpat_s, start_H, widthH)

    print(epiH.shape, epiV.shape)

    center_view = nlf[3, 3, :, :, :]
    cv2.rectangle(center_view, (fixSpat_t, start_W), (fixSpat_t+2, start_W+widthW), color=(255,0,0), thickness=1)
    cv2.rectangle(center_view, (start_H, fixSpat_s), (start_H+widthH, fixSpat_s+2), color=(0,0,255), thickness=1)
    
    show_all_3(center_view, epiV, epiH.transpose(1,0,2))
# # Horizontal EPIs:
# fixSpat = 300
# fixSpat2 = 300
# epi = get_epi(plf, 0, 3, fixSpat)
# epiV = get_epi(plf, 1, 3, fixSpat2)
# print(epi.shape, epiV.shape)    

# center_view = nlf[3, 3, :, :, :]
# cv2.rectangle(center_view, (0,fixSpat), (nlf.shape[3],fixSpat), color=(255,0,0), thickness=2)
# cv2.rectangle(center_view, (fixSpat2,0), (fixSpat2,nlf.shape[2]), color=(0,0,255), thickness=2)

# plt.axis('off')
# plt.imshow(center_view)
# plt.savefig("epi_Central_1.5_SD.png", transparent=True)
# plt.close()

# # plt.axis('off')
# plt.imshow(epi.transpose(1,0,2))
# plt.savefig("epiH.png", transparent=True)
# plt.close()

# # plt.axis('off')
# plt.imshow(epiV.transpose(1,0,2))
# plt.savefig("epiV.png", transparent=True)
# plt.close()






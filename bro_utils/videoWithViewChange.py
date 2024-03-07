#!/usr/bin/env python3

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from plenpy.lightfields import LightField
from moviepy.editor import *
from tqdm.auto import tqdm



def read_lf2(path_to_lf):
   npyLF = np.load(path_to_lf)
   npyLF = npyLF.transpose(0,2,3,1)
   compatible_npy = np.zeros((7, 7, 256, 192, 3))
   # compatible_npy = np.zeros((7, 7, 480, 640, 3))
   for u in range(7):
      for v in range(7):
            # print(u, v)
            compatible_npy[u, v, :, :, :] = npyLF[v+(u*7), :, :, :]
         
   plenLF = LightField(compatible_npy)
   return plenLF, compatible_npy


def show_lf2(plenLF):
   plenLF.show()


def get_input_disparity2(path_to_disp):
    pass


def do_viewSynthesis(npyLF):
    imgSeq = []
    u_samples = [0,0,0,0,0,0,0,1,2,3,4,5,6,6,6,6,6,6,6,5,4,3,2,1,0]
    v_samples = [0,1,2,3,4,5,6,6,6,6,6,6,6,5,4,3,2,1,0,0,0,0,0,0,0]
    for i in range(len(u_samples)):
        imgSeq.append(npyLF[u_samples[i], v_samples[i], : , :, :])
    return imgSeq


if __name__ == '__main__':
    VID_PATH = "./Video_2/"
    LFS = [ f"{VID_PATH}LF_{i}.npy" for i in range(1, len(os.listdir(VID_PATH))+1)]

    imgs = []
    for i, lf_name in tqdm(enumerate(LFS)):
        if i < 291 or i > 321:
            continue
        plf, nlf = read_lf2(lf_name)
        imgs.append(nlf[2,0,:,:,:])
        if (i+1) % 50 == 0:
            imgs.extend(do_viewSynthesis(nlf))
    
    print("Total imgs:", len(imgs))
    clip = ImageSequenceClip([imgs[i]*255 for i in range(len(imgs))], fps=5)
    clip.write_gif(f'Video_withViewSynth.gif', fps=5)
    # clip.write_videofile(f'Video_2_withViewSynth.mp4', fps=10)
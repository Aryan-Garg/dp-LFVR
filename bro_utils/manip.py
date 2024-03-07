#!/usr/bin/env python3

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from plenpy.lightfields import LightField
from moviepy.editor import *
from tqdm.auto import tqdm


def read_lf(path_to_lf, lf_shape=(7,7,480,640,3)):
   npyLF = np.load(path_to_lf)
   npyLF = npyLF.transpose(0,2,3,1)
   # print(npyLF.shape, np.amax(npyLF), np.amin(npyLF))
   compatible_npy = np.zeros(lf_shape)
   # compatible_npy = np.zeros((7, 7, 480, 640, 3))

   for u in range(7):
      for v in range(7):
            # print(u, v)
            compatible_npy[u, v, :, :, :] = npyLF[v+(u*7), :, :, :]
         
   plenLF = LightField(compatible_npy)
   return plenLF, compatible_npy


def show_lf(plenLF):
   plenLF.show()


def get_disp(plenLF):
    disp, conf = plenLF.get_disparity(method='structure_tensor', fusion_method='tv_l1', epi_method='2.5d')
    return disp, conf


# plf, nlf = read_lf(f"./baseline_3_OurRes/2x_LF_13.npy")
# show_lf(plf)
# print(nlf[3,6,...].shape)
# cv2.imwrite("2x_extreme_view.png", cv2.cvtColor(nlf[3,6,...]*255, cv2.COLOR_BGR2RGB))
# img = plf.get_refocus(0.5, aperture='disk')

# plt.imshow(img)
# plt.show()
# for i in tqdm(range(1,15)):
#    plf, nlf = read_lf(f"baseline_3_OurRes/LF_9.npy")
#    #  show_lf(plf)
#    refocuses = np.arange(-1,1.01, 0.09)
#    imgs = plf.get_refocus(refocuses, aperture='hann_rotational')
#    clip = ImageSequenceClip([imgs[:,:,:,i]*255 for i in range(len(refocuses))], fps=5)
#    clip.write_gif(f'hann_rotational_aperture.gif', fps=5)


# plf.show_refocus_interactive()
# disp, conf = get_disp(pLF)
# fig = plt.figure(figsize=(10, 10))
# for i in range(7):
#     for j in range(7):
#         # print(i*7 + j + 1)
#         plt.subplot(7, 7, i*7 + j + 1 )
#         plt.imshow(nlf[i*7 + j, :, :, :])
#         plt.axis('off')
# plt.savefig("model_out.png", transparent=True, bbox_inches='tight', pad_inches=0)
# plt.show()
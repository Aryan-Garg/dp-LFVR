#!/usr/bin/env python

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from plenpy.lightfields import LightField
from moviepy.editor import *
from tqdm.auto import tqdm
import manip as utils


def normalize(img):
    return (img - np.amin(img)) / (np.amax(img) - np.amin(img))


def shift_and_add(nlf, mask, alpha=1, counter=-1):
    # Alpha: 1 --> No shift
    assert counter > 0, "Counter not provided!"

    img = np.zeros((nlf.shape[2], nlf.shape[3], nlf.shape[4]))
    for x in tqdm(range(img.shape[0])):
        for y in range(img.shape[1]):
            for i in range(nlf.shape[0]):
                for j in range(nlf.shape[1]):
                    if mask[i][j]:
                        shift_fx = (i-3) * ((alpha-1)/alpha)
                        shift_fy = (3-j) * ((alpha-1)/alpha)
                        img[x][y] += (1./counter) * nlf[i,j, 
                                                        min(int(shift_fx + x), nlf.shape[2]-1), 
                                                        min(int(shift_fy + y), nlf.shape[3]-1), 
                                                        :]

    return img / (alpha**2)


def get_mask(mask_type=None, radius=2, square_side=2):
    assert mask_type in [None, 'circle', 'square', 'heart', 'plus', 'cross']

    mask = np.zeros((7,7))

    if mask_type == None:
        print("Using Full Aperture!\nNo mask type provided")
        return np.ones((7,7))
    elif mask_type == 'circle':
        # print("Using Circle Aperture!")
        mask = np.zeros((7,7))
        couter = 0
        for i in range(7):
            for j in range(7):
                if (i-3)**2 + (j-3)**2 <= radius**2:
                    mask[i,j] = 1
                    couter += 1

        return mask, couter
    
    elif mask_type == 'square':
        # print("Using Square Aperture!")
        counter = 0
        for i in range(7):
            for j in range(7):
                if abs(i-3) <= square_side and abs(j-3) <= square_side:
                    mask[i,j] = 1
                    counter += 1

        return mask, counter
    
    elif mask_type == 'cross':
        # print("Using Cross Aperture!")
        mask[3][3] = 1
        # horizontal
        mask[3][2] = 1
        mask[3][4] = 1
        # vertical
        mask[2][3] = 1
        mask[4][3] = 1
        mask[5][3] = 1
        return mask, 7
    
    elif mask_type == 'plus':
        # print("Using Plus Aperture!")
        mask[3][3] = 1
        # horizontal
        mask[3][2] = 1
        mask[3][4] = 1
        # vertical
        mask[2][3] = 1
        mask[4][3] = 1
        return mask, 5
    
    elif mask_type == 'heart':
        # print("Using Heart Aperture!")
        mask[1][2] = 1
        mask[1][4] = 1
        for i in range(1,6):
            mask[2][i] = 1
            mask[3][i] = 1
        for i in range(2,5):
            mask[4][i] = 1
        mask[5][3] = 1

        return mask, 16
    else:
        print("Not Using Any Aperture!\nExiting...")
        exit()


def swirly_per_pixel_mask(): # fish-eye transformation
    mask = np.zeros((7,7,480,640,3))
    pass

# TODO 1:
# Aperture shape simulation
# TODO 2:
# Aperture size simulation

if __name__ == '__main__':
    plf, nlf = utils.read_lf("./baseline_1.5_SD/LF_4.npy", (7,7,480,640,3))
    mask, counter = get_mask("circle", radius=1)
    
    alphas = np.arange(0.8, 1.7, 0.4)
    imgs = []
    for alpha in alphas:
        imgs.append(shift_and_add(nlf, mask, alpha, counter=counter))
    
    clip = ImageSequenceClip([imgs[i]*255 for i in range(len(imgs))], fps=10)
    clip.write_gif(f'vary_alpha_wide_circle.gif', fps=10)
    
    # print(saImg.shape, np.amax(saImg), np.amin(saImg))
    # plt.imshow(saImg)
    # plt.show()
    # radii = [1,2,3,4]
    # alpha = [1.2,1.2,1.2,1.2]
    # for i in tqdm(range(len(radii))):
    #     mask = get_mask("circle", radius=radii[i])
    #     saImg = shift_and_add(nlf, mask, alpha=alpha[i])
    #     plt.axes(False)
    #     plt.title(f"Aperture: Circle(r={radii[i]}), Alpha: {alpha[i]}")
    #     plt.savefig(f"radius_{radii[i]}_alph_1.2.png", saImg, bbox_inches='tight')
    #     plt.show()
    #     plt.close()

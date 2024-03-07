#!/usr/bin/env python


import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import cv2
import bro_utils as bruh


def make_grid_from_np_array(arr, np_arr_names):
    """
    np_arr: 4D array of shape (N, height, width, channels)
    """
    
    # make matplotlib grid
    for ii in range(int(len(arr)/8) + 1):
        fig, ax = plt.subplots(nrows=8, ncols=8, figsize=(10, 10))
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        ax = ax.ravel() 

        # plot images
        for i in range(8*ii, 8*ii + 64):
            ax[i-(8*ii)].imshow(arr[i])
            ax[i-(8*ii)].set_axis_off()
            ax[i-(8*ii)].set_title(f"{np_arr_names[i].split('/')[-2]}")

        # super title
        fig.suptitle(f"{np_arr_names[0].split('/')[-3]}", fontsize=16)

        # save
        plt.show()
        plt.savefig(f"{ii+1}_{np_arr_names[0].split('/')[-3]}.png", dpi=300)
        plt.close()


def save_center(np_arr, np_arr_names):
    for i in range(len(np_arr)):
        plt.imshow(np_arr[i])
        plt.axis('off')
        plt.show()
        plt.savefig(f"./FbyF_temp_results/center/{np_arr_names[i].split('/')[-2]}.png")
        plt.close()


def get_all_lf_paths_for(dataset):
    lf_paths = []
    for root, _, files in os.walk(f"./FbyF_temp_results/{dataset}_Seq/"):
        for file in files:
            if file.endswith(".npy"):
                lf_paths.append(os.path.join(root, file))
    return lf_paths


if __name__ == '__main__':
    datasets = ["Kalantari"] # "Hybrid", "Kalantari", "TAMULF"
    for ds in tqdm(datasets):
        lf_paths = get_all_lf_paths_for(ds)
        # print(ds, len(lf_paths))
        np_arr = []
        for lf in lf_paths:
            _, nlf = bruh.read_lf(lf, (7,7,256,192,3))
            np_arr.append(nlf[3,3,...])

        save_center(np_arr, lf_paths)
        

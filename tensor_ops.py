import torch
import cv2
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.autograd import Variable
import imageio
import layers

def imtensor2imnp(tensor):
    '''
    This will be of the size [N,3,h,w]
    returns a list of N images
    '''
    imnp = tensor.data.cpu().numpy()
    imnp = imnp.transpose(0,2,3,1)
    list_im = [imnp[k,...] for k in range(imnp.shape[0])]
    return list_im

def lftensor2lfnp(lftensor):
    '''
    This will be of the size [N,V,3,h,w]
    returns a list of N LF images
    '''
    imnp = lftensor.data.cpu().numpy()
    imnp = imnp.transpose(0,1,3,4,2)
    list_im = [imnp[k,...] for k in range(imnp.shape[0])]
    return list_im

def save_video_from_lf(lf_img, save_lf_path):
    h,w = lf_img.shape[-3:-1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        save_lf_path, fourcc, 5, (w, h))
    for k in range(len(lf_img)):
        #print(lf_img[k, ..., ::-1].shape)
        out.write(np.uint8(lf_img[k, ..., ::-1] * 255))
    out.release()

def save_img(img, path):
    h,w = lf_img.shape[-2:]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        save_lf_path, fourcc, 5, (w, h))
    for k in range(len(lf_img)):
        out.write(np.uint8(lf_img[k, ..., ::-1] * 255))
    out.release()

def get_paths(save_path, step, N):
    # get lf image paths
    lf_paths = []
    img_paths = []
    for k in range(N):
        lf_paths.append(os.path.join(save_path, f'{step*N+k:02d}_pred_lf.mp4'))
        img_paths.append(os.path.join(save_path, f'{step*N+k:02d}_img.jpg'))
    return lf_paths, img_paths
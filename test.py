#!/usr/bin/env python
import argparse
import os
import sys
import uuid
from datetime import datetime as dt
import json
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import random
warnings.filterwarnings('ignore')

import cv2
import time

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
import torchvision

import model_io
import models
from vid_dataloader import LFDataLoader
from loss import *
from utils import RunningAverage, RunningAverageDict, denormalize3d, denormalize1d
import tensor_ops as tensor_utils
import imageio

# from lpips_pytorch import LPIPS
from loss import SSIMLoss
from flopth import flopth

# set all seeds to 42
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


class Tester():
    def __init__(self, args):
        self.epoch = 0
        self.args = args
        self.refinement = args.refine
        #################################### Setup GPU device ######################################### 
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu') # 0

        if args.val_height >= 480 or args.val_width >= 640: # too big for our 11.9 GiB GPU
            self.device = torch.device('cpu')

        print('device: %s' % self.device)
        
        ######################################## Model ################################################
        # number of predictions for TD
        td_chans = self.args.rank*self.args.num_layers*3
        in_chans = 5
        if self.args.disp_input:
            in_chans += 1
        self.model = models.Gamma.build(in_chans=in_chans,
                                        layers=self.args.num_layers,
                                        rank=self.args.rank,
                                        temporal=self.args.temporal,
                                        dp_cross_attention=False)
        if args.flops:
            flops, params = flopth(self.model.cpu(), 
                                   inputs=(torch.rand(1, 5,args.val_height, args.val_width).cpu(),), 
                                   show_detail=True)
            
            print(flops, params)
            exit()

        self.model.to(self.device)
        # self.model.encoder.original_model.conv_stem = models.Conv2dSame(5, 48, kernel_size=(3, 3), stride=(2, 2), bias=False)
        
        path = os.path.join(args.root, args.exp_name, 'checkpoints', args.checkpoint)
        checkpoint = torch.load(path, map_location='cpu')['model']
        self.model = model_io.load_weights(self.model, checkpoint)
        self.model = self.model.to(self.device)

        ######################################## Tensor Display ################################################
        self.val_td_model = models.multilayer(height=args.val_height, width=args.val_width, 
                                              args=self.args, device=self.device)

        if self.refinement:
            self.ref_model = models.RefinementBlock(patch_size=1)
            checkpoint = torch.load('/data2/aryan/mono-eccv/weights/eccv_refine_net.pt', map_location='cpu')['model']
            self.ref_model.load_state_dict(checkpoint)
            self.ref_model = self.ref_model.to(self.device)


        if self.args.get_model_size:
            print("Main model size: ")
            self.get_size_in_MB(self.model)
            print("TD model size: ")
            self.get_size_in_MB(self.val_td_model)
            exit()
        
        ####################################### Save test results ##############################################
        self.save_path = os.path.join(args.results, args.exp_name)
        os.makedirs(self.save_path, exist_ok=True)


    def get_size_in_MB(self, model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2

        print('model size: {:.3f}MB'.format(size_all_mb))
        total_params = int(sum(p.numel() for p in model.parameters()))
        print('total_params: {:.3f}'.format(total_params))

        return size_all_mb, total_params


    def calculate_psnr(self, img1, img2):
        with torch.no_grad():
            img1 = 255*img1#.cpu()
            img2 = 255*img2#.cpu()
            mse = torch.mean((img1 - img2)**2)
            if mse == 0:
                return float('inf')
            return 20 * math.log10(255.0 / math.sqrt(mse))
        

    def calculate_ssim(self, img1, img2):
        with torch.no_grad():
            ssim = SSIMLoss()
            N, V, C, H, W = img1.shape
            img1 = img1.reshape(N*V, C, H, W).cpu()
            img2 = img2.reshape(N*V, C, H, W).cpu()
            loss = 1 - 0.1*ssim(img1, img2)
            return loss


    def test(self, test_loader, max_disp, zero_plane):
        ###############################################################################################
        self.model.eval()

        ###############################################################################################
        # some globals
        iters = len(test_loader)
        # interpolate = nn.Upsample(size=(1080, 720), mode='bilinear')
        if self.args.calcMetrics:
            psnr_avg_1 = RunningAverage()
            ssim_avg_1 = RunningAverage()
            
            f = open(os.path.join(self.save_path, f'{self.args.dataset}_res_{self.args.val_height}x{self.args.val_width}.txt'), 'w')
        
        if self.args.calcTime:
            time_f = open(os.path.join(self.save_path, f'TIME_{self.args.dataset}_res_{self.args.val_height}x{self.args.val_width}.txt'), 'w')
            all_times = []

        with torch.no_grad():
            with tqdm(enumerate(test_loader), total=len(test_loader), 
                      desc=f"Testing {self.args.dataset}") as vepoch:
                
                for i, batch in vepoch:                       
                    video_length = len(batch)
                    prev_state = None

                    pred_lfs = []
                    orig_imgs = []
                    
                    if self.args.calcMetrics:
                        ids = sorted(batch.keys())
                        leng = len(ids) - 1
                        gt_lfs = []
                        psnrs_1 = []
                        ssims_1 = []
                        # print("ids:", ids)
                        for id in ids[1:-1]:
                            disp = -1 * (batch[id]['disp'] - zero_plane) * max_disp

                            if self.args.disp_input:
                                curr_img = torch.cat([batch[id]['rgb'], 
                                                      batch[id]['left_dp'], 
                                                      batch[id]['right_dp'], 
                                                      disp],dim=1).to(self.device)
                            else:
                                curr_img = torch.cat([batch[id]['rgb'], 
                                                      batch[id]['left_dp'], 
                                                      batch[id]['right_dp']],dim=1).to(self.device)
                            # prev_img = torch.cat([batch[id-1]['rgb'], batch[id-1]['left_dp'], batch[id-1]['right_dp']],dim=1).to(self.device)
                            # next_img = torch.cat([batch[id+1]['rgb'], batch[id+1]['left_dp'], batch[id+1]['right_dp']],dim=1).to(self.device)
                            # NOTE: prev and next will be used for video input methods

                            curr_orig_image = batch[id]['rgb'].cpu()
                            orig_imgs.append(denormalize3d(curr_orig_image).cpu().numpy())
                            # prev_orig_image = batch[id-1]['rgb']
                            # next_orig_image = batch[id+1]['rgb']


                            img = curr_img # NOTE: Frame Input 
                            
                            if self.args.calcTime:
                                starttime = time.time()
                                decomposition, depth_planes, _ = self.model(img, prev_state)
                                pred_lf = self.val_td_model(decomposition, depth_planes)
                                if self.refinement:
                                    lf_inp = torch.cat([pred_lf, img.unsqueeze(1)], dim=1)
                                    mask, corr_lf = self.ref_model(lf_inp)
                                    ref_lf = mask*corr_lf + (1-mask)*pred_lf
                                    ref_lf = ref_lf.clip(0, 1)
                                    pred_lf = ref_lf.cpu()
                                elapsed_time = time.time() - starttime
                                time_f.write(str(elapsed_time)+"\n")
                                all_times.append(elapsed_time)
                                pred_lf = pred_lf.cpu()
                            else:
                                decomposition, depth_planes, _ = self.model(img, prev_state)
                                pred_lf = self.val_td_model(decomposition, depth_planes)
                                if self.refinement:
                                    lf_inp = torch.cat([pred_lf, batch[id]['rgb'].unsqueeze(1).to(self.device)], dim=1).to(self.device)
                                    mask, corr_lf = self.ref_model(lf_inp)
                                    ref_lf = mask*corr_lf + (1-mask)*pred_lf
                                    ref_lf = ref_lf.clip(0, 1)
                                    pred_lf = ref_lf.cpu()
                                pred_lf = pred_lf.cpu()
                            # pred_lf.clip(0, 1, out=pred_lf)
                            
                            if self.args.save_results:
                                saveInd = str(i).zfill(4)
                                os.makedirs(os.path.join(self.save_path, f"{self.args.val_height}x{self.args.val_width}"), exist_ok=True)
                                self.save_lf(pred_lf, os.path.join(self.save_path, f"{self.args.val_height}x{self.args.val_width}", f'{saveInd}.npy'))
                                # print(orig_imgs[-1].shape)
                                cv2.imwrite(os.path.join(self.save_path,f"{self.args.val_height}x{self.args.val_width}", f'{saveInd}.png'), orig_imgs[-1][0].transpose(1,2,0) * 255.)

                            pred_lfs.append(pred_lf)

                            if self.args.calcMetrics:
                                gt_lf = batch[id]['lf'].cpu()
                                gt_lfs.append(gt_lf)
                                # print(f"Shape check {self.args.dataset}: {pred_lf.shape}, {gt_lf.shape}")
                                pred_psnr_1 = self.calculate_psnr(pred_lf, gt_lf)
                                pred_ssim_1 = self.calculate_ssim(pred_lf, gt_lf)
                                # print("psnr, ssim :: ", pred_psnr_1, pred_ssim_1)
                                psnr_avg_1.append(pred_psnr_1)
                                ssim_avg_1.append(pred_ssim_1)
                                psnrs_1.append(pred_psnr_1)
                                ssims_1.append(pred_ssim_1)
                    
                        avg_psnr_1 = sum(psnrs_1)/len(psnrs_1)
                        avg_ssim_1 = sum(ssims_1)/len(ssims_1)
                        string = 'Sample {0:2d} => Init. PSNR: {1:.4f}, Init. SSIM: {2:.4f}\n'.format(i, avg_psnr_1, avg_ssim_1)
                        f.write(string)
                        vepoch.set_postfix(init_psnr=f"{psnr_avg_1.get_value():0.2f}({avg_psnr_1:0.2f})",
                                       init_ssim=f"{ssim_avg_1.get_value():0.2f}({avg_ssim_1:0.2f})")
                    
                    else:
                        img = batch['center'].to(self.device)
                        orig_img = denormalize3d(img, self.device).cpu().numpy()
                        print(orig_img.shape)
                        left_pd = batch['left_pd']
                        right_pd = batch['right_pd']

                       
                        left_pd = left_pd.to(self.device)
                        right_pd = right_pd.to(self.device)

                        inp = torch.cat([img, left_pd, right_pd], axis=1)
                        # Pass None into dummy temporal state variable if temporal=False

                        decomposition, depth_planes, _ = self.model(inp, None)
                        pred_lf = self.val_td_model(decomposition, depth_planes)
                        # pred_lfs.append(pred_lf.cpu())
                        # orig_imgs.append(orig_img.cpu())
                        if self.refinement:
                            lf_inp = torch.cat([pred_lf, img.unsqueeze(1)], dim=1)
                            mask, corr_lf = self.ref_model(lf_inp)
                            ref_lf = mask*corr_lf + (1-mask)*pred_lf
                            ref_lf = ref_lf.clip(0, 1)
                            pred_lf = ref_lf.cpu()


                        if self.args.save_results:
                            saveInd = str(i).zfill(4)
                            os.makedirs(os.path.join(self.save_path, f"LFs"), exist_ok=True)
                            self.save_lf(pred_lf, os.path.join(self.save_path, f'{self.args.dataset}_LF_{saveInd}.npy'))
                            cv2.imwrite(os.path.join(self.save_path, f'{saveInd}.png'), orig_img.cpu().numpy())

                if self.args.calcMetrics:
                    pred_avg_psnr = psnr_avg_1.get_value()
                    pred_avg_ssim = ssim_avg_1.get_value()
                    string = '\n\n---------\nAverage PSNR: {0:.4f}\nAverage SSIM: {1:.4f}\n---------'.format(pred_avg_psnr, pred_avg_ssim)
                    f.write(string)
                    f.close()

                if self.args.calcTime:
                    avg_time = sum(all_times) / len(all_times)
                    time_f.write("Avg time: " + str(avg_time) + "\n")
                    time_f.close()


    def save_lf(self, lf, path):
        np.save(path, lf)


    def main_worker(self):
        test_loader = LFDataLoader(self.args, 'test', calcMetrics=self.args.calcMetrics).data
        self.test(test_loader, self.args.baseline_factor, 0.3)


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)



if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='Testing script. Default values of all arguments are recommended for reproducibility', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    ####################################### Experiment arguments ######################################
    parser.add_argument('--root', default='results/', type=str, help='directory with save checkpoints')
    parser.add_argument('-exp', '--exp_name', default='AddLN_GA16_RGB+DP_30@21-Aug_17:59:46', type=str, help='name of the experiment')
    parser.add_argument('-ckpt', '--checkpoint', default='093.pt', type=str, help='epochs stored weight to use')
    
    parser.add_argument('--results', default='metrics/', type=str, help='directory to save results')
    
    parser.add_argument('-sr', '--save_results', default=False, action='store_true', help='save results')
    
    parser.add_argument('--gpu', default=1, type=int, help='which gpu to use')
    parser.add_argument('--workers', default=1, type=int, help='number of workers for data loading')
    
    ######################################## Dataset parameters #######################################
    parser.add_argument('--dataset', default='TAMULF', type=str, help='Dataset to train/test on')

    parser.add_argument('--data_path', default='/data2/raghav/datasets/Pixel4_3DP/rectified', type=str,
                        help='path to images') # Won't be used if metrics are to be calculated. lf_path will be used instead
    
    # '/data/prasan/datasets/LF_datasets/DPT-depth' | Unimatch BA: './disparity_maps/disp_pixel4_BA'
    parser.add_argument('--disp_path', default='media/data/prasan/datasets/LF_video_datasets/DPT-depth', type=str,
                        help='path to disparity') # won't be used but need to pass for the dataloaders' sake
    parser.add_argument('--disp_input', default=False, action='store_true', help='use disparity as input')
    # Our generated/synthetifilenames_file_evalc dual pixels from LFs
    parser.add_argument('--genDP_path', default='/data2/aryan/Hybrid/test_dp', type=str, help='path to generated dual pixels') 

    # Pixel4: './train_inputs/full_Pixel4/test_files.txt'
    parser.add_argument('--filenames_file_eval', default='./FT_train-test_files/Selected_Seq/test_files.txt',
                        type=str, help='path to the filenames text file for online evaluation')

    ############################################# I/0 parameters ######################################
    parser.add_argument('-vh', '--val_height', type=int, help='validate height', default=352) # training hxw: 256 x 192
    parser.add_argument('-vw', '--val_width', type=int, help='validate width', default=528) 
    # LIC: 352 x 528 | Srinivasan: 188 x 270 (D) | Li: 192x192 | ours: 256x192 | 480p or SD or 480x640 
    # | Mono-lfvr: 176x264
    parser.add_argument('-cc', '--color_corr', default=True, action='store_true', help='color correct images')

    ##################################### Learning parameters #########################################
    parser.add_argument('-bs', '--batchsize', default=1, type=int, help='batch size')
    
    ##################################### Tensor Display parameters #########################################
    parser.add_argument('--rank', default= 12, type=int, help='rank of the tensor display')
    parser.add_argument('--num_layers', default= 3, type=int, help='number of layers in the tensor display')
    parser.add_argument('--angular', default=7, type=int, help='number of angular views to output')
    parser.add_argument('-tdf', '--td_factor', default=1, type=int, help='disparity factor for layers')

    # parser.add_argument('--eigen_crop', help='if set, crops according to Eigen NIPS14', action='store_true')
    # parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
    # parser.add_argument('--do_kb_crop', help='Use kitti benchmark cropping', action='store_true')

    parser.add_argument('--calcMetrics', help='if set, calculates metrics', action='store_true', default=False)
    parser.add_argument('--calcTime', action='store_true', default=False)
    parser.add_argument('--lf_path', default='/data/prasan/datasets/LF_datasets/', type=str, help='path to light field dataset')
    parser.add_argument('-ty', '--type', type=str, default='resize', 
                        help='whether to train with crops or resized images')
    
    # /media/data/prasan/datasets/LF_datasets/DPT-depth/ | /data2/aryan/unimatch/dp_otherDS/
    parser.add_argument('--otherDS_disp_path', default='/data2/aryan/unimatch/dp_otherDS/', type=str)
    parser.add_argument('--get_model_size', action='store_true', default=False)

    parser.add_argument('--temporal', action='store_true', default=False)

    parser.add_argument('--flops', action='store_true', default=False)

    parser.add_argument('--refine', action='store_true', default=False)
    parser.add_argument('-bf', '--baseline_factor', default=1.5, type=float, help='scale max disparity by this factor')

    args = parser.parse_args()

    if args.results != '.' and not os.path.isdir(args.results):
        os.makedirs(args.results)

    tester = Tester(args)
    tester.main_worker()
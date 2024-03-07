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
from dataloader import LFDataLoader
from loss import *
from utils import RunningAverage, RunningAverageDict, denormalize3d, denormalize1d
import tensor_ops as tensor_utils
import imageio

import lpips
import wandb

# set all seeds to 42
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

class DST_Conv(nn.Module):
    def __init__(self, dim_embed):
        super().__init__()
        self.convBlock = nn.Sequential(*[
            nn.Conv2d(dim_embed, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        ])

    def forward(self, x):
        return self.convBlock(x)


class downstream_disp():
    def __init__(self, args):
        self.start_epoch = 1
        self.mode = args.mode
        self.args = args
        self.log_step = args.log_step
        #################################### Setup GPU device ######################################### 
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu') # 0

        if args.val_height >= 480 or args.val_width >= 640: # too big for our 11.9 GiB GPU
            self.device = torch.device('cpu')

        print('device: %s' % self.device)
        
        ######################################## Model ################################################
        # number of predictions for TD
        td_chans = 12*3*3
        in_chans = 5
        self.model = models.Gamma.build(in_chans=in_chans,
                                        layers=self.args.num_layers,
                                        rank=self.args.rank,
                                        temporal=False,
                                        dp_cross_attention=False)
        
        self.model.to(self.device)
        root=os.path.join(self.args.results, self.args.resume_id, 'checkpoints', self.args.checkpoint)
        checkpoint = torch.load(root, map_location='cpu')['model']
        self.model = model_io.load_weights(self.model, checkpoint)
        # print(self.model)
        self.model = self.model.to(self.device)

        ######################################## Tensor Display ################################################
        self.val_td_model = models.multilayer(height=args.val_height, width=args.val_width, 
                                              args=self.args, device=self.device)

        self.DST_disp_layers = DST_Conv(dim_embed = 3*self.args.angular*self.args.angular)
        self.DST_disp_layers.to(self.device)

        ######################################## Loss ################################################
        self.criterionL1 = nn.L1Loss().to(self.device)
        self.criterionL2 = nn.MSELoss().to(self.device)
        self.criterionLPIPS = lpips.LPIPS(net='vgg').to(self.device)
        
        ######################################## Optimizer ################################################
        self.optimizer = optim.AdamW(self.DST_disp_layers.parameters())
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                            T_max=100, eta_min=0.00001, last_epoch=-1)
        
        ######################################## Logging ################################################
        run_id = '{}'.format(dt.now().strftime('%d-%h_%H:%M:%S'))
        self.name = '{}@{}'.format(args.exp_name, run_id)
        print('Logging directory: {}/{}'.format(args.results, self.name))
        wandb.login()
        wandb.tensorboard.patch(root_logdir=os.path.join(args.results, self.name))
        wandb.init(sync_tensorboard=True, 
                       config=args, 
                       project="lfvr",
                       name=args.exp_name) 
        self.writer = SummaryWriter(os.path.join(args.results, self.name))
        with open('{}/{}/config.json'.format(args.results, self.name), 'w+') as file:
            json.dump(args.__dict__, file, indent=4)
        print('Configs dumped in {}/{}/config.json'.format(args.results, self.name))

        self.save_path = os.path.join(args.results, args.exp_name)
        os.makedirs(self.save_path, exist_ok=True)

        ######################################## Train Help ################################################
        self.epoch = 0


    def compute_loss(self, disp_pred, disp_gt):
        loss = self.criterionL1(disp_pred, disp_gt)
        loss_mse = self.criterionL2(disp_pred, disp_gt)
        loss_lpips = self.criterionLPIPS.forward(disp_pred, disp_gt).mean()
        total_loss = 0.8 * loss + 0.2 * loss_mse + loss_lpips
        return total_loss


    def train(self, train_loader):
        self.model.eval()
        self.val_td_model.eval()
        self.DST_disp_layers.train()

        iters = len(train_loader)
        step = self.epoch * iters
        best_loss = np.inf

        with tqdm(enumerate(train_loader), total=len(train_loader), 
                  desc=f"Downstream-Disp") as vepoch:
            self.writer.add_scalar("epoch", self.epoch, self.epoch)
            this_epoch_loss = 0
            for i, batch in vepoch:                                      
                dispGT = batch['disp'].to(self.device)
                # print("dispGT:", dispGT.shape)
                curr_img = torch.cat([batch['center'], 
                                        batch['left_pd'], 
                                        batch['right_pd']],dim=1).to(self.device)
                
                curr_orig_image = batch['center'].cpu()
        
                img = curr_img 
                decomposition, depth_planes, _ = self.model(img, None)
                pred_lf = self.val_td_model(decomposition, depth_planes)
                pred_lf = pred_lf.reshape((1, 3*self.args.angular*self.args.angular, self.args.val_height, self.args.val_width))
                disp_pred = self.DST_disp_layers(pred_lf)
                self.optimizer.zero_grad()
                loss = self.compute_loss(disp_pred, dispGT)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                this_epoch_loss += loss.item()
                if step % self.args.log_step == 0:
                    self.writer.add_scalar(f'Train/LR', self.scheduler.get_last_lr(), step)
                    self.writer.add_scalar(f'Train/Loss', loss.item(), step)
                    img = torchvision.utils.make_grid(curr_orig_image)
                    self.writer.add_image(f'Train/Image', img, step)
                    left_pd = torchvision.utils.make_grid(batch['left_pd'].cpu())
                    self.writer.add_image(f'Train/Left DP', left_pd, step)
                    right_pd = torchvision.utils.make_grid(batch['right_pd'].cpu())
                    self.writer.add_image(f'Train/Right DP', right_pd, step)
                    uni_disp = torchvision.utils.make_grid(dispGT.cpu())
                    self.writer.add_image('Train/GT(Uni)Disparity', uni_disp, step)
                    pred_disp = torchvision.utils.make_grid(disp_pred.cpu())
                    self.writer.add_image('Train/Pred-Disparity', pred_disp, step)
                step += 1

            this_epoch_loss /= len(train_loader)

            if best_loss > this_epoch_loss:
                best_loss = this_epoch_loss
                torch.save(self.DST_disp_layers.state_dict(), f'results/DST_disp/{self.epoch}.pth')
                print("Best train loss seen:", loss.item())

            vepoch.set_description(f"Ep: {self.epoch+1} | Loss: {loss.item():.3f}")        
    

    def save_disp(self, disp, path):
        toSave = np.uint8(disp*255)
        np.save(path, toSave)


    def main_worker(self):
        train_loader = LFDataLoader(self.args, 'train').data
        val_loader = LFDataLoader(self.args, 'eval').data
        # test_loader = LFDataLoader(self.args, 'test').data

        if self.mode == 'train':
            self.start_epoch = 0
            for epoch in range(self.start_epoch, self.args.epochs):
                self.epoch = epoch
                self.train(train_loader)
                # if epoch % self.args.validate_every == 0:
                #     # max_disp_list = [1.5, 2.0]
                #     for max_disp in max_disp_list:
                #         self.validate(val_loader, max_disp, zero_plane)


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='Training script. Default values of all arguments are recommended for reproducibility', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    ####################################### Experiment arguments ######################################
    parser.add_argument('-exp', '--exp_name', default='DST_disp', type=str, help='name of the experiment')
    parser.add_argument('-d', '--dry_run', default=False, action='store_true', help='choose to save checkpoints')
    parser.add_argument('--mode', default='train', type=str, help='train/test option')


    parser.add_argument('--log_step', default=1000, type=int, help='gaps to log training steps')
    parser.add_argument('--val_log_step', '-vls', default=120, type=int, help='gaps to log validation steps')
    parser.add_argument('-r', '--resume', default=True, action='store_true', help='Resume from checkpoint')
    parser.add_argument('-rid', '--resume_id', default="AddLN_GA16_RGB+DP_30@21-Aug_17:59:46", type=str, help='experiment to resume')
    parser.add_argument('-c', '--checkpoint', default="093.pt", type=str, help='checkpoint to resume from')
    parser.add_argument('--validate_every', default=1, type=int, help='validate once every k epoch')
    
    parser.add_argument('--test_after_full_train', default=False, action='store_true', help='test after training')
    
    parser.add_argument('--results', default='results', type=str, help='directory to save results')
    
    parser.add_argument('--gpu', default="1", type=str, help='which gpu(s) to use')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loading')
  
    ######################################## Dataset parameters #######################################
    parser.add_argument('--datapath', default='/data2/raghav/datasets/Pixel4_3DP/rectified', type=str,
                        help='path to dataset')
    parser.add_argument('--unrect_datapath', default='/data2/raghav/datasets/Pixel4_3DP/unrectified', type=str,
                        help='path to dataset')
    
    # Unimatch disparity maps ----------------------------------------------------------
    # NOTE: Default: True
    parser.add_argument('--unimatch_disp_path', '-udp', default='./disparity_maps/disp_pixel4_BA', type=str,
                        help='path to disparity maps from unimatch')
    parser.add_argument('--use_unimatch', '-uud', default=True, action='store_true')
    # -----------------------------------------------------------------------------------

    parser.add_argument('--filenames_file_folder',
                        default='/data2/aryan/lfvr/train_inputs/Pixel4_3DP_frame_skip10',
                        type=str, help='path to the folder containing filenames to use')

    ############################################# Model I/O parameters ######################################
    parser.add_argument('-th', '--train_height', type=int, help='train height', default=256)
    parser.add_argument('-tw', '--train_width', type=int, help='train width', default=192)
    parser.add_argument('-vh', '--val_height', type=int, help='validate height', default=256)
    parser.add_argument('-vw', '--val_width', type=int, help='validate width', default=192)

    ##################################### Learning parameters #########################################
    parser.add_argument('-e', '--epochs', default=30, type=int, help='number of total epochs to run')
    parser.add_argument('-bs', '--batchsize', default=1, type=int, help='batch size')
  
    ##################################### Tensor Display parameters #########################################
    parser.add_argument('-m', '--rank', default=12, type=int, help='rank of the tensor display')
    parser.add_argument('-l', '--num_layers', default=3, type=int, help='number of layers in the tensor display')
    parser.add_argument('-v', '--angular', default=7, type=int, help='number of angular views to output')
    parser.add_argument('-tdf', '--td_factor', default=1, type=int, help='disparity factor for layers')

    #### Finetuning & Metrics
    parser.add_argument('--dataset', default='Pixel4', type=str, help='Dataset to train on')
    parser.add_argument('--genDP_path', default='/data2/aryan/Stanford/test_dp', type=str, help='path to generated dual pixels') 
    parser.add_argument('--calcMetrics', help='if set, calculates metrics', action='store_true', default=False)
    parser.add_argument('--lf_path', default='/media/data/prasan/datasets/LF_video_datasets/', type=str, help='path to light field dataset')
    parser.add_argument('-ty', '--type', type=str, default='resize', 
                        help='whether to train with crops or resized images')
    parser.add_argument('-cc', '--color_corr', default=True, action='store_true', help='color correct images')
    parser.add_argument('--otherDS_disp_path', default='/data2/aryan/unimatch/dp_otherDS/', type=str, help='path to other datasets disparity maps')
    parser.add_argument('--model_summary', default=False, action='store_true', help='print model summary')
    parser.add_argument('--finetune', default=False, action='store_true', help='finetune on other datasets')
    parser.add_argument('--visualization_shuffle', default=False, action='store_true', help='shuffle images for visualization')


    args = parser.parse_args()

    if args.results != '.' and not os.path.isdir(args.results):
        os.makedirs(args.results)

    trainer = downstream_disp(args)
    trainer.main_worker()
    wandb.finish()
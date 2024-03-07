#!/usr/bin/env python

### Base imports ###
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
# NOTE:Always keep it at 0; 1,2 & 3 don't have the right drivers
from datetime import datetime as dt
import json
import numpy as np
import random
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

### Torch imports ###
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
import torchvision

import wandb

### Custom imports ###
import model_io
import models
from vid_dataloader import LFDataLoader
from loss import *
from utils import RunningAverage, denormalize3d, denormalize1d

### Summarization ###
from torchsummary import summary
from torchviz import make_dot, make_dot_from_trace

# print("[+] Custom imports successful")

### WandB ###
import wandb

# print("[+] All imports successful\n")

class Trainer():
    def __init__(self, args):
        self.args = args
        self.mode = args.mode
        #################################### Setup GPU device ######################################### 
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device(f'cpu')
        print('Device: %s' % self.device)
        
        ######################################## Model ################################################
        
        in_channels = 15
        if args.depth_input:
            in_channels += 1

        self.model = models.Gamma.build(in_chans=in_channels,
                                        layers=self.args.num_layers,
                                        rank=self.args.rank,
                                        temporal=args.temporal)
        
        if args.depth_input:
            self.model.encoder.original_model.conv_stem = models.Conv2dSame(10, 48, kernel_size=(3, 3), 
                                                                            stride=(2, 2), bias=False)
        #checkpoint = torch.load('GoPro-REDS/trial@31-Jan_13:40:57/checkpoints/024.pt'.format(args.rank), 
        #                        map_location='cpu')['model']
        #self.model = model_io.load_weights(self.model, checkpoint)
        self.model = self.model.to(self.device)

        ##################################### Tensor Display ##########################################
        self.train_td_model = models.multilayer(height=args.train_height, width=args.train_width, 
                                                args=self.args, device=self.device)
        
        self.val_td_model = models.multilayer(height=args.val_height, width=args.val_width, 
                                              args=self.args, device=self.device)
        self.disp_list = [0.6, 1.2, 1.8, 2.4]
        # Overfit on one disparity
        # self.disp_list = [1.5]

        ########################################## Losses #############################################
        self.photometric_loss = PhotometricConsistency(args, self.device)
        self.geometric_loss = GeometricConsistency(args, self.device)
        if self.args.temporal:
            self.temporal_loss = TemporalConsistency(args, self.device)
        self.occlusion_loss = OcclusionLoss(args, self.device, ratio=args.ratio)
        self.smoothness_loss = SmoothLoss(args, self.device)
        self.bins_loss = BinsChamferLoss()
        
        ######################################### Optimizer ###########################################
        params = self.model.parameters()
        self.optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
        
        if self.args.model_summary:
            with open('model_summary_temporal.txt', 'w+') as f:
                f.write("Print Model:\n---------------------------------------------\n"+str(self.model))
                f.write('\n\n')
                f.write("Print Train TD Model:\n------------------------------------\n"+str(self.train_td_model))
                f.write('\n\n')
                f.write("Print Val TD Model:\n---------------------------------------\n"+str(self.val_td_model))
                f.write('\n------------------------------------------------------------------------------------\n\n')
        

                f.write("Model Summary:\n--------------------------------------")
                f.write(str(summary(self.model, (15, 256, 192), device=self.device)))
                f.write('\n\n')
                f.write("Train TD Model Summary:\n")
                
                input_data = torch.randn(1, 3, 12, 3, 256, 192).to(self.device)
                other_input_data = torch.randn(1, 3).to(self.device)

                f.write(str(summary(self.train_td_model, input_data, other_input_data, device=self.device)))
                f.write('\n\n')
                f.write("Val TD Model Summary:\n")
                f.write(str(summary(self.val_td_model, input_data, other_input_data, device=self.device)))
                f.write('\n\n')
                f.write("------------------------------------------------------------------------------------\n")

                f.write("Model Layer Names:\n--------------------------------------\n")
                for nam, param in self.model.named_parameters():
                    f.write("{}: {}\n".format(nam, param.shape))
                f.write('\n\n')

                f.write("Train TD Model Layer Names:\n--------------------------------------\n")
                for nam, param in self.train_td_model.named_parameters():
                    f.write("{}: {}\n".format(nam, param.shape))
                f.write('\n\n')

                f.write("Val TD Model Layer Names:\n--------------------------------------\n")
                for nam, param in self.val_td_model.named_parameters():
                    f.write("{}: {}\n".format(nam, param.shape))
                f.write('\n\n')
                f.write("\nBass aur nahi hai...\n")
                exit()


        ###################################### Logging setup ##########################################
        if not args.dry_run:
            if not args.mode == 'test':
                print('Training {}'.format(args.exp_name))

            run_id = '{}'.format(dt.now().strftime('%d-%h_%H:%M:%S'))
            self.name = '{}@{}'.format(args.exp_name, run_id)
            print('Logging directory: {}/{}'.format(args.results, self.name))

            # NOTE: Must call this before creating a Tensorboard SummaryWriter
            wandb.login() # env variable WANDB_API_KEY must be set in your environment or manually enter!
            
            wandb.tensorboard.patch(root_logdir=os.path.join(args.results, self.name))
            wandb.init(sync_tensorboard=True, 
                       config=args, 
                       project="lfvr",
                       name=args.exp_name) 
            # ProTip: Remove name = "args.exp_name" so that wandb gives wonderful exp names ;)

            
            ### Tensorboard logging ###
            self.writer = SummaryWriter(os.path.join(args.results, self.name))
            with open('{}/{}/config.json'.format(args.results, self.name), 'w+') as file:
                json.dump(args.__dict__, file, indent=4)
            print('Configs dumped in {}/{}/config.json'.format(args.results, self.name))


    def compute_disp(self, depth):
        min_disp = 1 / self.args.max_depth
        max_disp = 1 / self.args.min_depth
        scaled_disp = 1 / depth
        disp = (scaled_disp - min_disp) / (max_disp - min_disp)
        return disp


    def compute_losses(self, imgs, disp, decomposition, pred_lf, depth_planes):
        
        for k,v in imgs.items():
            imgs[k] = v.to(self.device)

        prev_img, img, next_img = imgs[-1], imgs[0], imgs[1]
        
        photo_loss = self.photometric_loss(img, pred_lf)
        
        geo_loss, warped_lf = self.geometric_loss(img, disp, pred_lf)

        temp_loss_inst = 0.0
        if self.args.temporal:
            temp_loss, warped_prev_lf = self.temporal_loss(img, prev_img, warped_lf)
            temp_loss_inst = self.args.temp_consistency * temp_loss

        smooth_loss = self.smoothness_loss(decomposition, disp)
        
        bins_loss = self.bins_loss(depth_planes, disp)
        
        H, W = img.shape[-2:]
        if H>450 or W>300 or self.args.occ_handling==0:
            occ_loss = 0.0
            logs = [None, None, None]
        else:
            occ_loss, logs = self.occlusion_loss(imgs, disp, pred_lf)
        
        total_loss = photo_loss + self.args.lf_consistency * geo_loss + \
                     self.args.occ_handling * occ_loss + self.args.w_chamfer * bins_loss + \
                     self.args.smoothness * smooth_loss + temp_loss_inst
        

        if self.args.temporal:
            logs.extend([warped_lf, warped_prev_lf]) 
        else:
            logs.extend([warped_lf, None]) # None for warped_prev_lf

        return total_loss, logs


    def train(self, train_loader):
        self.model.train()
        self.optimizer.zero_grad()
        # Globals:
        iters = len(train_loader)
        step = self.epoch * iters
        best_loss = np.inf
        loss_avg = RunningAverage()

        ################################# Train loop ##################################################
        torch.autograd.set_detect_anomaly(True)
        with tqdm(enumerate(train_loader), total=len(train_loader), 
                      desc='Epoch: {}/{}. Loop: Train'.format(self.epoch+1, self.args.epochs)) as tepoch:
            self.writer.add_scalar("epoch", self.epoch+1, self.epoch+1)      
            for i, batch in tepoch:
                max_disp = random.choice(self.disp_list)
                zero_plane = random.uniform(0.2, 0.4)
                video_length = len(batch)
                prev_state = None

                for id in range(1, video_length-1):
                    # self.optimizer.zero_grad()
                    # print(batch[id]['rgb'].shape, batch[id]['left_dp'].shape, batch[id]['right_dp'].shape)
                    curr_img = torch.cat([batch[id]['rgb'], batch[id]['left_dp'], batch[id]['right_dp']], dim=1).to(self.device)
                    prev_img = torch.cat([batch[id-1]['rgb'], batch[id-1]['left_dp'], batch[id-1]['right_dp']], dim=1).to(self.device)
                    next_img = torch.cat([batch[id+1]['rgb'], batch[id+1]['left_dp'], batch[id+1]['right_dp']], dim=1).to(self.device)

                    curr_orig_image = batch[id]['rgb']
                    curr_orig_image_log = denormalize3d(curr_orig_image)
                    prev_orig_image = batch[id-1]['rgb']
                    prev_orig_image_log = denormalize3d(prev_orig_image)
                    next_orig_image = batch[id+1]['rgb']
                    next_orig_image_log = denormalize3d(next_orig_image)

                    # curr_orig_image = curr_orig_image.to(self.device)
                    # prev_orig_image = prev_orig_image.to(self.device)
                    # next_orig_image = next_orig_image.to(self.device)

                    dpt_disp = batch[id]['disp'].to(self.device)
                    disp = -1 * (batch[id]['disp'].to(self.device) - zero_plane) * max_disp
                    
                    if self.args.depth_input:
                        img = torch.cat([prev_img, curr_img, next_img, disp], dim=1)
                    else:
                        img = torch.cat([prev_img, curr_img, next_img], dim=1)

                    decomposition, depth_planes, state = self.model(img, prev_state)
                    pred_lf = self.train_td_model(decomposition, depth_planes)

                    #viz_td = torch.mean(decomposition, dim=2, keepdim=False)
                    #N, layer, C, H, W = viz_td.shape
                    #maxi = torch.max(torch.max(torch.max(torch.max(viz_td, dim=1, keepdim=True).values, 
                    #       dim=2, keepdim=True).values, dim=3, keepdim=True).values, 
                    #       dim=4, keepdim=True).values
                    #viz_td = viz_td/maxi
                    #viz = viz_td.view(N*layer, C, H, W)

                    imgs = {0:curr_orig_image, 
                            -1:prev_orig_image, 
                            1:next_orig_image}
                    
                    loss, logs = self.compute_losses(imgs, disp, decomposition, pred_lf, depth_planes)
                    loss = loss / self.args.accumulation_steps
                    loss.backward()

                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    if (i+1) % self.args.accumulation_steps == 0 or (i+1) == len(train_loader):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    
                    loss_avg.append(loss.item())

                    if state:
                        prev_state = [state[0].detach(), state[1].detach()]

                tepoch.set_postfix(total_loss=f"{loss_avg.get_value():0.4f}({loss.item():0.4f})")
                
                if not self.args.dry_run and step % self.args.log_step == 0:
                    self.writer.add_scalar(f'Train/Total-Loss', loss.item(), step)
                    curr_img = torchvision.utils.make_grid(curr_orig_image_log)
                    self.writer.add_image(f'Train/Current-Image', curr_img, step)
                    prev_img = torchvision.utils.make_grid(prev_orig_image_log)
                    self.writer.add_image(f'Train/Previous-Image', prev_img, step)
                    next_img = torchvision.utils.make_grid(next_orig_image_log)
                    self.writer.add_image(f'Train/Next-Image', next_img, step)
                    dpt_disp = torchvision.utils.make_grid(dpt_disp.detach().cpu())
                    self.writer.add_image('Train/DPT-Disparity', dpt_disp, step)
                    scld_disp = torchvision.utils.make_grid(torch.abs(disp))
                    self.writer.add_image('Train/Scaled-Disparity', scld_disp, step)
                    #viz = torchvision.utils.make_grid(viz, nrow=self.args.num_layers)
                    #self.writer.add_image('Train/TD layer vizualization', viz, step)
                    for j in range(min(4,self.args.batchsize)):
                        self.writer.add_video(f"Train/Predicted LF-{j}", pred_lf[j:j+1,...].clip(0, 1).detach().cpu(), step)

                        forward_warp_lf, warped_lfs, mask, warped_lf, warped_prev_lf = logs
                        
                        self.writer.add_video(f"Train/Warped LF-{j}", warped_lf[j:j+1,...].clip(0, 1).detach().cpu(), step)
                        if warped_prev_lf is not None: # When OCC was first added, temp loss was not used
                            self.writer.add_video(f"Train/Warped prev LF-{j}", warped_prev_lf[j:j+1,...].clip(0, 1).detach().cpu(), step)
                        
                        '''
                        warped_lfs[-1] = warped_lfs[-1].reshape(pred_lf.shape)
                        warped_lfs[1] = warped_lfs[1].reshape(pred_lf.shape)
                        mask = mask.reshape(pred_lf.shape)
                        forward_warp_lf = forward_warp_lf.reshape(pred_lf.shape)
                        self.writer.add_video(f"Train/prev_warped_lf/{j}",warped_lfs[-1][j:j+1,...], step)
                        self.writer.add_video(f"Train/next_warped_lf/{j}",warped_lfs[1][j:j+1,...], step)
                        self.writer.add_video(f"Train/mask/{j}",mask[j:j+1,...], step)
                        self.writer.add_video(f"Train/forward_warp_lf/{j}",forward_warp_lf[j:j+1,...], step)
                        '''

                step += 1
                self.scheduler.step()


    def validate(self, val_loader, max_disp, zero_plane):
        ###############################################################################################
        self.model.eval()

        ###############################################################################################
        # some globals
        iters = len(val_loader)
        step = self.epoch * iters
        loss_avg = RunningAverage()

        ################################# Validation loop #############################################
        with torch.no_grad():
            if not self.args.dry_run:
                model_io.save_checkpoint(self.model, self.optimizer, self.scheduler, 
                                         self.epoch, '{0:03d}.pt'.format(self.epoch),
                                         root=os.path.join(self.args.results, self.name, 'checkpoints'))
                
            with tqdm(enumerate(val_loader), total=len(val_loader), 
                      desc='Epoch: {}/{}. Loop: Validation-{}, {}'.format(self.epoch+1, self.args.epochs,
                      max_disp, zero_plane)) as vepoch:
                
                for i, batch in vepoch:
                    video_length = len(batch)
                    prev_state = None

                    for id in range(1,video_length-1):
                        curr_img = torch.cat([batch[id]['rgb'], batch[id]['left_dp'], batch[id]['right_dp']],dim=1).to(self.device)
                        prev_img = torch.cat([batch[id-1]['rgb'], batch[id-1]['left_dp'], batch[id-1]['right_dp']],dim=1).to(self.device)
                        next_img = torch.cat([batch[id+1]['rgb'], batch[id+1]['left_dp'], batch[id+1]['right_dp']],dim=1).to(self.device)


                        curr_orig_image = batch[id]['rgb']
                        curr_orig_image_log = denormalize3d(curr_orig_image)
                        prev_orig_image = batch[id-1]['rgb']
                        prev_orig_image_log = denormalize3d(prev_orig_image)
                        next_orig_image = batch[id+1]['rgb']
                        next_orig_image_log = denormalize3d(next_orig_image)

                        # curr_orig_image = curr_orig_image.to(self.device)
                        # prev_orig_image = prev_orig_image.to(self.device)
                        # next_orig_image = next_orig_image.to(self.device)

                        dpt_disp = batch[id]['disp'].to(self.device)

                        disp = -1 * (batch[id]['disp'].to(self.device) - zero_plane) * max_disp
                        if self.args.depth_input:
                            img = torch.cat([prev_img, curr_img, next_img, disp], dim=1)
                        else:
                            img = torch.cat([prev_img, curr_img, next_img], dim=1)

                        decomposition, depth_planes, state = self.model(img, prev_state)
                        pred_lf = self.val_td_model(decomposition, depth_planes)

                        #viz_td = torch.mean(decomposition, dim=2, keepdim=False)
                        #N, layer, C, H, W = viz_td.shape
                        #maxi = torch.max(torch.max(torch.max(torch.max(viz_td, dim=1, keepdim=True).values, 
                        #    dim=2, keepdim=True).values, dim=3, keepdim=True).values, dim=4, keepdim=True).values
                        #viz_td = viz_td/maxi
                        #viz = viz_td.view(N*layer, C, H, W)
                        
                        imgs = {0:curr_orig_image, 
                                -1:prev_orig_image, 
                                1:next_orig_image}
                        
                        loss, logs = self.compute_losses(imgs, disp, decomposition, pred_lf, depth_planes)
                        
                        loss_avg.append(loss.item())

                        prev_state = state

                    vepoch.set_postfix(total_loss=f"{loss_avg.get_value():0.4f}({loss.item():0.4f})")
                    if step % self.args.log_step == 0 and not self.args.dry_run:
                        
                        self.writer.add_scalar(f'Val-{max_disp}, {zero_plane}/Loss', loss, step)
                        curr_img = torchvision.utils.make_grid(curr_orig_image_log)
                        self.writer.add_image(f'Val-{max_disp}, {zero_plane}/Current Image', curr_img, step)
                        prev_img = torchvision.utils.make_grid(prev_orig_image_log)
                        self.writer.add_image(f'Val-{max_disp}, {zero_plane}/Previous Image', prev_img, step)
                        next_img = torchvision.utils.make_grid(next_orig_image_log)
                        self.writer.add_image(f'Val-{max_disp}, {zero_plane}/Next Image', next_img, step)
                        dpt_disp = torchvision.utils.make_grid(dpt_disp)
                        self.writer.add_image(f'Val-{max_disp}, {zero_plane}/DPT Disparity', dpt_disp.detach().cpu(), step)
                        scld_disp = torchvision.utils.make_grid(torch.abs(disp))
                        self.writer.add_image(f'Val-{max_disp}, {zero_plane}/Scaled Disparity', scld_disp, step)
                        #viz = torchvision.utils.make_grid(viz, nrow=self.args.num_layers)
                        #self.writer.add_image(f'Val-{max_disp}, {zero_plane}/TD layer vizualization', viz, step)
                        for j in range(min(4,self.args.batchsize)):
                            self.writer.add_video(f"Val-{max_disp}, {zero_plane}/Predicted LF-{j}", pred_lf[j:j+1,...].clip(0, 1).detach().cpu(), step)

                            forward_warp_lf, warped_lfs, mask, warped_lf, warped_prev_lf = logs
                        
                            self.writer.add_video(f"Val-{max_disp}, {zero_plane}/Warped-LF-{j}", warped_lf[j:j+1,...].clip(0, 1).detach().cpu(), step)
                            if warped_prev_lf is not None: # When OCC was first added, temp loss was not used
                                self.writer.add_video(f"Val-{max_disp}, {zero_plane}/Warped-Prev-LF-{j}", warped_prev_lf[j:j+1,...].clip(0, 1).detach().cpu(), step)
                            
                            #warped_lfs[-1] = warped_lfs[-1].reshape(pred_lf.shape)
                            #warped_lfs[1] = warped_lfs[1].reshape(pred_lf.shape)
                            #mask = mask.reshape(pred_lf.shape)
                            #forward_warp_lf = forward_warp_lf.reshape(pred_lf.shape)
    
    
    def test(self, test_loader, max_disp, zero_plane):
        ###############################################################################################
        self.model.eval()

        ###############################################################################################
        # some globals
        iters = len(test_loader)
        step = self.epoch * iters
        loss_avg = RunningAverage()

        with torch.no_grad():
            with tqdm(enumerate(test_loader), total=len(test_loader), 
                      desc='Epoch: {}/{}. Loop: Validation-{}, {}'.format(self.epoch+1, self.args.epochs,
                      max_disp, zero_plane)) as vepoch:
                
                for i, batch in vepoch:
                    video_length = len(batch)
                    prev_state = None

                    for id in range(1,video_length-1):
                        curr_img = torch.cat([batch[id]['rgb'], batch[id]['left_dp'], batch[id]['right_dp']],dim=1).to(self.device)
                        prev_img = torch.cat([batch[id-1]['rgb'], batch[id-1]['left_dp'], batch[id-1]['right_dp']],dim=1).to(self.device)
                        next_img = torch.cat([batch[id+1]['rgb'], batch[id+1]['left_dp'], batch[id+1]['right_dp']],dim=1).to(self.device)


                        curr_orig_image = batch[id]['rgb']
                        curr_orig_image_log = denormalize3d(curr_orig_image)
                        prev_orig_image = batch[id-1]['rgb']
                        prev_orig_image_log = denormalize3d(prev_orig_image)
                        next_orig_image = batch[id+1]['rgb']
                        next_orig_image_log = denormalize3d(next_orig_image)

                        # curr_orig_image = curr_orig_image.to(self.device)
                        # prev_orig_image = prev_orig_image.to(self.device)
                        # next_orig_image = next_orig_image.to(self.device)

                        dpt_disp = batch[id]['disp'].to(self.device)

                        disp = -1 * (batch[id]['disp'].to(self.device) - zero_plane) * max_disp
                        if self.args.depth_input:
                            img = torch.cat([prev_img, curr_img, next_img, disp], dim=1)
                        else:
                            img = torch.cat([prev_img, curr_img, next_img], dim=1)

                        decomposition, depth_planes, state = self.model(img, prev_state)
                        pred_lf = self.val_td_model(decomposition, depth_planes)

                        #viz_td = torch.mean(decomposition, dim=2, keepdim=False)
                        #N, layer, C, H, W = viz_td.shape
                        #maxi = torch.max(torch.max(torch.max(torch.max(viz_td, dim=1, keepdim=True).values, 
                        #    dim=2, keepdim=True).values, dim=3, keepdim=True).values, dim=4, keepdim=True).values
                        #viz_td = viz_td/maxi
                        #viz = viz_td.view(N*layer, C, H, W)
                        
                        imgs = {0:curr_orig_image, 
                                -1:prev_orig_image, 
                                1:next_orig_image}
                        
                        loss, logs = self.compute_losses(imgs, disp, decomposition, pred_lf, depth_planes)
                        
                        loss_avg.append(loss.item())

                        prev_state = state

                    vepoch.set_postfix(total_loss=f"{loss_avg.get_value():0.4f}({loss.item():0.4f})")
                    if step % self.args.log_step == 0 and not self.args.dry_run:
                        
                        self.writer.add_scalar(f'Test-{max_disp}, {zero_plane}/Loss', loss, step)
                        curr_img = torchvision.utils.make_grid(curr_orig_image_log)
                        self.writer.add_image(f'Test-{max_disp}, {zero_plane}/Current Image', curr_img, step)
                        prev_img = torchvision.utils.make_grid(prev_orig_image_log)
                        self.writer.add_image(f'Test-{max_disp}, {zero_plane}/Previous Image', prev_img, step)
                        next_img = torchvision.utils.make_grid(next_orig_image_log)
                        self.writer.add_image(f'Test-{max_disp}, {zero_plane}/Next Image', next_img, step)
                        dpt_disp = torchvision.utils.make_grid(dpt_disp)
                        self.writer.add_image(f'Test-{max_disp}, {zero_plane}/DPT Disparity', dpt_disp.detach().cpu(), step)
                        scld_disp = torchvision.utils.make_grid(torch.abs(disp))
                        self.writer.add_image(f'Test-{max_disp}, {zero_plane}/Scaled Disparity', scld_disp, step)
                        #viz = torchvision.utils.make_grid(viz, nrow=self.args.num_layers)
                        #self.writer.add_image(f'Val-{max_disp}, {zero_plane}/TD layer vizualization', viz, step)
                        for j in range(min(4,self.args.batchsize)):
                            self.writer.add_video(f"Test-{max_disp}, {zero_plane}/Predicted LF-{j}", pred_lf[j:j+1,...].clip(0, 1).detach().cpu(), step)
                            
                            if self.args.isConf:
                                np.save(f"{self.args.conf_save}LF_{i+1:04d}.npy", 
                                        pred_lf.detach().cpu().numpy(), 
                                        allow_pickle=True)
                            
                            forward_warp_lf, warped_lfs, mask, warped_lf, warped_prev_lf = logs
                        
                            self.writer.add_video(f"Test-{max_disp}, {zero_plane}/Warped-LF-{j}", warped_lf[j:j+1,...].clip(0, 1).detach().cpu(), step)
                            if warped_prev_lf is not None: # When OCC was first added, temp loss was not used
                                self.writer.add_video(f"Test-{max_disp}, {zero_plane}/Warped-Prev-LF-{j}", warped_prev_lf[j:j+1,...].clip(0, 1).detach().cpu(), step)
                            
                            #warped_lfs[-1] = warped_lfs[-1].reshape(pred_lf.shape)
                            #warped_lfs[1] = warped_lfs[1].reshape(pred_lf.shape)
                            #mask = mask.reshape(pred_lf.shape)
                            #forward_warp_lf = forward_warp_lf.reshape(pred_lf.shape)


    def main_worker(self):
        ###############################################################################################
        
        ##################################### Dataloaders #############################################
        train_loader = LFDataLoader(self.args, 'train').data
        val_loader = LFDataLoader(self.args, 'eval').data
        test_loader = LFDataLoader(self.args, 'test').data

        ###################################### Scheduler ##############################################

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                             max_lr=self.args.lr,
                                                             steps_per_epoch=len(train_loader),
                                                             epochs=self.args.epochs,
                                                             pct_start=0.1,)
        
        ############################### Check to resume training ######################################
        if self.mode == 'train':
            self.start_epoch = 0
            if args.resume and args.resume_id is not None:
                print("Trying to resume...")
                self.name = args.resume_id
                if os.path.isdir(os.path.join(args.results, args.resume_id)):
                    print('Logging directory: {}/{}'.format(args.results, args.resume_id))

                    directory = os.path.join(args.results, args.resume_id, 'checkpoints')
                    try:    
                        checkpoints = sorted(os.listdir(directory))
                        if checkpoints:
                            latest = checkpoints[-1]
                            print("[train.py@308] Latest ckpt found:", latest)
    
                            self.model, self.optimizer, self.scheduler, self.start_epoch = model_io.load_checkpoint(
                                    os.path.join(directory, latest), self.model, self.optimizer, self.scheduler, 
                                    replace=False, total_epochs=args.epochs, lenTrainLoader=len(train_loader))

                            print('Resuming training from epoch {} of {} experiment'.format(self.start_epoch,
                                                                                            args.resume_id))
                    except Exception as e:
                        print("Error:", e)
                        print("FAILED to resume. Starting from scratch.")
                        self.start_epoch = 0

                    self.writer = SummaryWriter(os.path.join(args.results, args.resume_id))

            elif args.resume:
                raise ValueError('Resume ID not given')

            for epoch in range(self.start_epoch, self.args.epochs):
                self.epoch = epoch
                self.train(train_loader)
                if epoch % self.args.validate_every == 0:
                    zero_plane = 0.3
                    # max_disp_list = [1.5, 2.0]
                    max_disp_list = [1.5]
                    for max_disp in max_disp_list:
                        self.validate(val_loader, max_disp, zero_plane)
            
            if args.test_after_full_train:
                for max_disp in [1.5]: # max_disp_list
                    self.test(test_loader, max_disp, zero_plane)

        elif self.mode == 'test':
            assert args.resume_id is not None, "Resume ID not given"
            print(f"[+] Trying to test using {args.resume_id}...")
            self.epoch = 0
            self.name = args.resume_id
            if os.path.isdir(os.path.join(args.results, args.resume_id)):
                print('Logging directory: {}/{}'.format(args.results, args.resume_id))

                directory = os.path.join(args.results, args.resume_id, 'checkpoints')
                try:    
                    checkpoints = sorted(os.listdir(directory))
                    if checkpoints:
                        latest = checkpoints[-1]
                        print("[train.py@308] Latest ckpt found:", latest)
    
                        self.model, self.optimizer, self.scheduler, self.start_epoch = model_io.load_checkpoint(
                                    os.path.join(directory, latest), self.model, self.optimizer, self.scheduler, 
                                    replace=True, total_epochs=args.epochs, lenTrainLoader=len(train_loader))

                        print('Testing using {} experiment'.format(args.resume_id))
                        self.model.to(self.device)
                        
                except Exception as e:
                    print("Error:", e)
                    print("[!] FAILED to Test")
                        

                self.writer = SummaryWriter(os.path.join(args.results, args.resume_id))

                max_disp_list = [1.5]
                zero_plane = 0.3
                for max_disp in max_disp_list:
                    self.test(test_loader, max_disp, zero_plane)

        else:
            print("Invalid mode. Choose from train/test")



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
    parser.add_argument('-exp', '--exp_name', default='Cont_Temp+Occ_skip3_100eps', type=str, help='name of the experiment')
    parser.add_argument('-d', '--dry_run', default=False, action='store_true', help='choose to save checkpoints')
    parser.add_argument('--mode', default='train', type=str, help='train/evaluate option')
    parser.add_argument('--log_step', default=50, type=int, help='gaps to log training and valdiation steps')
    parser.add_argument('-r', '--resume', default=False, action='store_true', help='Resume from checkpoint')
    parser.add_argument('-rid', '--resume_id', default=None, type=str, help='experiment to resume')
    parser.add_argument('--validate_every', default=1, type=int, help='validate once every k epoch')
    
    parser.add_argument('--test_after_full_train', default=True, action='store_true', help='test after training')
    
    parser.add_argument('--results', default='results', type=str, help='directory to save results')
    
    parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loading')
    
    ######################################## Dataset parameters #######################################
    parser.add_argument('--dataset', default='TAMULF', type=str, help='Dataset Name')

    parser.add_argument('--datapath', default='/data2/raghav/datasets/Pixel4_3DP/rectified', type=str,
                        help='path to dataset')
    parser.add_argument('--unrect_datapath', default='/data2/raghav/datasets/Pixel4_3DP/unrectified', type=str,
                        help='path to dataset')
    parser.add_argument('-acc', '--accumulation_steps', default=16, type=int, help='gradient accumulation steps')
    
    parser.add_argument('--unimatch_disp_path', '-udp', default='./disparity_maps/disp_pixel4_BA', type=str,
                        help='path to disparity maps from unimatch')
    parser.add_argument('--use_unimatch', '-uud', default=True, action='store_true')
    parser.add_argument('--filenames_file_folder',
                        default='./train_inputs/Pixel4_3DP_skip3/',
                        type=str, help='path to the folder containing filenames to use')
    
    parser.add_argument('--visualization_shuffle', default=False, action='store_true', help='visualize input data')
    
    parser.add_argument('--isConf', default=False, action='store_true', help='whether to save complete LF for conference')
    parser.add_argument('--conf_save', default='./conference_results/', type=str, help='path to save conference results')
    
    # parser.add_argument('--filenames_file',
    #                     default='./train_inputs/Pixel4_3DP_skip10/train_files.json',
    #                     type=str, help='path to the filenames text file')
    # parser.add_argument('--filenames_file_eval',
    #                     default='./train_inputs/Pixel4_3DP_skip10/val_files.json',
    #                     type=str, help='path to the filenames text file for validation')
    
    ############################################# I/0 parameters ######################################
    parser.add_argument('-th', '--train_height', type=int, help='train height', default=256)
    parser.add_argument('-tw', '--train_width', type=int, help='train width', default=192)
    parser.add_argument('-vh', '--val_height', type=int, help='validate height', default=256) 
    parser.add_argument('-vw', '--val_width', type=int, help='validate width', default=192)
    parser.add_argument('--depth_input', default=False, action='store_true', 
                        help='whether to use depth as input to network')

    ######################################### Model parameters ########################################
    parser.add_argument('-lb', '--large_baseline', default=False, action='store_true', 
                        help='whether to use large baseline for training (CA)')
    parser.add_argument('--temporal', default=False, action='store_true', help='Enable ConvLSTM for temporal learning!')
    
    ####################################### Loss parameters ###########################################
    parser.add_argument('-wgt', '--edge_weight_mask', default=False, action='store_true', 
                        help='whether to use edge weights mask')
    parser.add_argument('-ss', '--ssim', default=False, action='store_true',
                        help='whether to use SSIM loss')
    parser.add_argument('-ws', '--w_ssim', default=0.5, type=float, help='weight value for SSIM loss')
    parser.add_argument('-lc', '--lf_consistency', default=1.0, type=float, 
                        help='weight value for geometric consistency')
    parser.add_argument('-tc', '--temp_consistency', default=0.2, type=float, 
                        help='weight value for temporal consistency')
    parser.add_argument('-oh', '--occ_handling', default=0.2, type=float, 
                        help='weight value for occlusion handling')
    parser.add_argument('-rt', '--ratio', default=0.2, type=float, 
                        help='ratio of angular view used for occlusion loss')
    parser.add_argument('-lt', '--loss_type', default='min', type=str, 
                        help='min/mean for occlusion loss')
    parser.add_argument('-sm', '--smoothness', default=0.1, type=float, 
                        help='weight value for light field geometric consistency')
    parser.add_argument('-wc', '--w_chamfer', default=2, type=float, help='weight value for chamfer loss')

    ####################################### RAFT parameters ###########################################
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    ####################################### CVT Params ###########################################
    parser.add_argument('-dpca', '--dp_cross_attention', default=False, action='store_true', 
                    help='Switch on cross attention')
    
    ##################################### Learning parameters #########################################
    parser.add_argument('-e', '--epochs', default=30, type=int, help='number of total epochs to run')
    parser.add_argument('-bs', '--batchsize', default=1, type=int, help='batch size')
    parser.add_argument('-lr', '--lr', default=5e-4, type=float, help='max learning rate')
    parser.add_argument('-wd', '--wd', default=5e-4, type=float, help='weight decay')
    
    ##################################### Tensor Display parameters #########################################
    parser.add_argument('-m', '--rank', default=12, type=int, help='rank of the tensor display')
    parser.add_argument('-l', '--num_layers', default=3, type=int, help='number of layers in the tensor display')
    parser.add_argument('-v', '--angular', default=7, type=int, help='number of angular views to output')
    parser.add_argument('-tdf', '--td_factor', default=1, type=int, help='disparity factor for layers')
    
    parser.add_argument('--otherDS_disp_path', default='/data2/aryan/unimatch/dp_otherDS/', type=str, help='path to other datasets disparity maps')
    parser.add_argument('--genDP_path', default='/data2/aryan/Stanford/test_dp', type=str, help='path to generated dual pixels') 
    parser.add_argument('--calcMetrics', help='if set, calculates metrics', action='store_true', default=False)
    parser.add_argument('--lf_path', default='/data/prasan/datasets/LF_datasets/', type=str, help='path to light field dataset')
    parser.add_argument('-ty', '--type', type=str, default='resize', 
                        help='whether to train with crops or resized images')
    parser.add_argument('-cc', '--color_corr', default=True, action='store_true', help='color correct images')
    parser.add_argument('--model_summary', default=False, action='store_true', help='print model summary')
    args = parser.parse_args()

    if args.results != '.' and not os.path.isdir(args.results):
        os.makedirs(args.results)

    trainer = Trainer(args)
    trainer.main_worker()
    wandb.finish()
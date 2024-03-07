#!/usr/bin/env python

### Base imports ###
import argparse
import os

# Set visible GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" 
# NOTE: Jarvis is a mis-matched cluster. Use 1,2,3 only == NVIDIA Titan X (Pascal) :( 
# 0 is GeForce GTX 1080 Ti

from datetime import datetime as dt
import json
import numpy as np
import random
from tqdm.auto import tqdm, trange
import time

import warnings
warnings.filterwarnings("ignore")

### Torch imports ###
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
import torchvision


### Summarization ###
from torchsummary import summary
from torchviz import make_dot, make_dot_from_trace


### Custom imports ###
import model_io
import models
from dataloader import LFDataLoader
from loss import *
from utils import RunningAverage, denormalize3d, denormalize1d
import lpips

import cv2

# print("[+] Custom imports successful")

### WandB ###
import wandb

# print("[+] All imports successful\n")


class Trainer():
    def __init__(self, args):

        self.args = args
        self.mode = args.mode
        self.ft_temp = args.ft_temp
        self.refinement = args.refinement
        #################################### Setup GPU device ######################################### 
        dev_ids = [int(x) for x in args.gpu.split(',')]
        if args.val_height >= 480:
            self.device = torch.device("cpu")
        else:
            for i in range(torch.cuda.device_count()):
                print(f"Available: id-{i} :: ", torch.cuda.get_device_properties(i).name)

            print(f"\nUsing GPUs: {args.gpu}\n------------------------------------------------------------")

            if len(dev_ids) == 1:
                self.device = torch.device(f"cuda:{dev_ids[0]}")
                if self.args.val_width == 640:
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
            print('Device(s): %s' % self.device)

        # NOTE: Comment out:
        # self.device = torch.device("cpu")
        
        ######################################## Model ################################################
        # number of predictions for TD
        in_channels = 5
        if args.depth_input:
            in_channels += 1
        
        if args.mono_ablation:
            in_channels = 4

        ### DataParrallel ###
        if torch.cuda.device_count() > 1 and len(dev_ids) > 1:
            self.model = nn.DataParallel(models.Gamma.build(in_chans=in_channels,
                                        layers=self.args.num_layers,
                                        rank=self.args.rank,
                                        temporal=False,
                                        dp_cross_attention=self.args.dp_cross_attention), 
                                        device_ids=dev_ids)
        else:
            self.model = models.Gamma.build(in_chans=in_channels,
                                        layers=self.args.num_layers,
                                        rank=self.args.rank,
                                        temporal=False,
                                        dp_cross_attention=self.args.dp_cross_attention)
        
        # print(self.model)
            
        self.model.to(self.device)

        ##################################### Tensor Display ##########################################
        self.train_td_model = models.multilayer(height=args.train_height, width=args.train_width, 
                                                args=self.args, device=self.device)
        
        self.val_td_model = models.multilayer(height=args.val_height, width=args.val_width, 
                                              args=self.args, device=self.device)
        
        # self.disp_list = [0.6, 1.2, 1.8, 2.4, 3.0]

        if self.args.model_summary:
            with open('model_summary.txt', 'w+') as f:
                f.write("Print Model:\n---------------------------------------------\n"+str(self.model))
                f.write('\n\n')
                f.write("Print Train TD Model:\n------------------------------------\n"+str(self.train_td_model))
                f.write('\n\n')
                f.write("Print Val TD Model:\n---------------------------------------\n"+str(self.val_td_model))
                f.write('\n------------------------------------------------------------------------------------\n\n')
        

                f.write("Model Summary:\n--------------------------------------")
                f.write(str(summary(self.model, (5, 256, 192), device=self.device)))
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

            # make_dot(self.model(torch.randn(1,5,256,192).to(self.device))[:-1], 
            #          params=dict(self.model.adaptive_bins_layer.named_parameters())).render("Gamma_AdaBins", format="pdf")
            
            # make_dot(self.train_td_model(input_data, other_input_data),
            #          params=dict(self.train_td_model.named_parameters())).render("train_td_model", format="jpg")
            
            # make_dot(self.val_td_model(input_data, other_input_data),
            #          params=dict(self.val_td_model.named_parameters())).render("val_td_model", format="jpg")

            exit()


        # NOTE: Overfit on one disparity
        self.disp_list = [1.5]

        ########################################## Losses #############################################
        if not self.mode == 'test':
            # self.stereo_photometric_loss = StereoPhotometricConsistency(args, self.device)
            self.photometric_loss = PhotometricConsistency(args, self.device)
            self.geometric_loss = GeometricConsistency(args, self.device)
            # self.smoothness_loss = SmoothLoss(args, self.device)
            self.bins_loss = BinsChamferLoss()
            self.best_val_loss = np.inf
            if self.args.ft_temp:
                self.temporal_loss = TemporalConsistency(args, self.device)
        
        self.refinement = args.refinement
        if self.refinement:
            self.ref_model = models.RefinementBlock(patch_size=1)
            checkpoint = torch.load('/data2/aryan/mono-eccv/weights/eccv_refine_net.pt', map_location='cpu')['model']
            self.ref_model.load_state_dict(checkpoint)
            self.ref_model = self.ref_model.to(self.device)
        # For fine-tuning
        # self.mse_loss = nn.MSELoss(reduction='mean')
        # self.lpips_loss = lpips.LPIPS(net='vgg').to(self.device)


        ######################################### Optimizer ###########################################
        params = self.model.parameters()
        self.optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
    
        ###################################### Logging setup ##########################################
        if not args.dry_run: # and not args.resume:
            if not args.mode == 'test':
                print('Training {}'.format(args.exp_name))

            run_id = '{}'.format(dt.now().strftime('%d-%h_%H:%M:%S'))
            self.name = '{}@{}'.format(args.exp_name, run_id)
            print('Logging directory: {}/{}'.format(args.results, self.name))

            ### WandB <3 ###
            # NOTE: Must call this before creating a Tensorboard SummaryWriter
            if args.mode != 'test':
                wandb.login() # env variable WANDB_API_KEY must be set in your environment or manually enter!
            
                wandb.tensorboard.patch(root_logdir=os.path.join(args.results, self.name))
                wandb.init(sync_tensorboard=True, 
                       config=args, 
                       project="lfvr",
                       name=args.exp_name) 
                # Remove name = "args.exp_name" so that wandb gives wonderful unicorn exp names ;)

            
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


    def compute_FT_losses(self, img, pred_lf, gt_lf=None):
        assert gt_lf is not None, "Ground truth LF not given for fine-tuning"

        mse_loss = self.mse_loss(pred_lf, gt_lf)

        # print(pred_lf.shape, gt_lf.shape)
        # lpips_loss = self.lpips_loss(pred_lf.view(-1, 3, 64, 64), gt_lf.view(-1, 3, 64, 64))
        # average all items in lpips_loss
        # lpips_loss = torch.mean(lpips_loss)
        # print("LPIPS loss:", lpips_loss)

        total_loss = mse_loss # + lpips_loss 

        return total_loss


    def compute_losses(self, img, disp, decomposition, pred_lf, depth_planes, right=None, left=None, nextRGB=None):
        photo_bool, geo_bool, bins_bool = True, True, True
        labl = self.args.loss_ablation
        if  labl != "bruh":
            assert labl in ["v1", "v2", "v3", "v4", "v5", "v6"]
            if labl == "v3":
                geo_bool, photo_bool = False, False
            elif labl == "v2":
                geo_bool, bins_bool = False, False
            elif labl == "v1":
                photo_bool, bins_bool = False, False
            elif labl == "v4":
                geo_bool = False
            elif labl == "v5":
                photo_bool = False
            elif labl == "v6":
                bins_bool = False
            else:
                print("Invalid loss ablation version\nPlus how did you avert the assertion?! Hacker!")
                exit()

        photo_loss = 0.
        if photo_bool:
            photo_loss = self.photometric_loss(img, pred_lf)
            # if right == None and left == None:
            #     photo_loss = self.photometric_loss(img, pred_lf)
            # else:
            #     photo_loss = self.stereo_photometric_loss(img, pred_lf, right, left)
                # print("Used stereo photometric loss")
                # old_photo_loss = self.photometric_loss(img, pred_lf)
                # print("Stereo Photo Loss:", photo_loss.item(), "| Old Photo Loss:", old_photo_loss.item())
        
        geo_loss = 0.
        if geo_bool:
            geo_loss, warped_lf = self.geometric_loss(img, disp, pred_lf)
        else:
            warped_lf = None

        # smooth_loss = self.smoothness_loss(decomposition, disp)
        
        temp_loss = 0.
        if self.ft_temp and nextRGB is not None:
            # NOTE: nextRGB is actually prev in vid sequence
            temp_loss, _ = self.temporal_loss(img, nextRGB, warped_lf)
            # print(temp_loss)
        
        bins_loss = 0.
        if bins_bool:
            bins_loss = self.bins_loss(depth_planes, disp) 
                
        total_loss = photo_loss + self.args.lf_consistency * geo_loss + self.args.w_chamfer * bins_loss # + \
                    #  self.args.smoothness * smooth_loss + self.args.temp_consistency * temp_loss
        
        return total_loss, warped_lf, temp_loss


    def train(self, train_loader):
        ###############################################################################################
        self.model.train()

        ###############################################################################################
        # some globals
        iters = len(train_loader)
        step = self.epoch * iters
        best_loss = np.inf
        loss_avg = RunningAverage()

        ################################# Train loop ##################################################
        torch.autograd.set_detect_anomaly(True)
        # print('\n\n\n')
        # print(type(train_loader))
        # print(next(iter(train_loader)))
        with tqdm(enumerate(train_loader), total=len(train_loader), 
                      desc='E{}/{}:T'.format(self.epoch+1, self.args.epochs),
                      ncols=100) as tepoch:
            self.writer.add_scalar("epoch", self.epoch+1, self.epoch+1)

            for i, batch in tepoch:
                max_disp = random.choice(self.disp_list)

                # print("iter:", i+1, " :")
                # print(batch)
                
                img = batch['center'].to(self.device)
                orig_img = denormalize3d(img, self.device)
                
                if "nextRGB" in batch:
                    nextRGB = batch['nextRGB'].to(self.device)
                else:
                    nextRGB = None
                # left = batch['left']
                # # print(left.shape)
                # if left.shape[-1] > 1:
                #     left = left.to(self.device)
                #     print("Got left")
                # else:
                #     left = None
                #     
                left = None
                right = None


                if not self.args.mono_ablation:
                    left_pd = batch['left_pd']
                    right_pd = batch['right_pd']

                    if self.args.rgb_only_ablation_experiment:
                        left_pd = torch.zeros_like(left_pd).to(self.device)
                        right_pd = torch.zeros_like(right_pd).to(self.device)
                    else:
                        left_pd = left_pd.to(self.device)
                        right_pd = right_pd.to(self.device)

                    orig_left_pd = denormalize1d(left_pd, self.device)
                    orig_right_pd = denormalize1d(right_pd, self.device)
                
                disp = batch['disp'].to(self.device) * max_disp
                
                # Dual pixel data stored after image
                # print(img.shape, left_pd.shape, right_pd.shape)
                # inp = torch.cat([img, left_pd, right_pd], axis=1)

                if self.args.depth_input:
                    inp = torch.cat([img, left_pd, right_pd, disp], dim=1)
                elif self.args.mono_ablation:
                    inp = torch.cat([img, disp], axis=1)
                else:
                    inp = torch.cat([img, left_pd, right_pd], axis=1)
                

                # Pass None into dummy temporal state variable if temporal=False
                decomposition, depth_planes, _ = self.model(inp, None)
                if self.args.adaptive_td_ablation:
                    depth_planes = torch.Tensor([-1., 0., 1.]).to(self.device).unsqueeze(0)
                # print(depth_planes.shape, depth_planes)

                pred_lf = self.train_td_model(decomposition, depth_planes)
                

                loss, logs, tempLoss = self.compute_losses(orig_img, disp, decomposition, pred_lf, depth_planes, right, left, nextRGB)
                loss = loss / self.args.accumulation_steps # To account for Gradient Accumulation
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                if (i+1) % self.args.accumulation_steps == 0 or (i+1) == len(train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                loss_avg.append(loss.item())

                tepoch.set_postfix(loss=f"{loss.item():0.4f}")
                if not self.args.dry_run and step % self.args.log_step == 0:
                    self.writer.add_scalar(f'Train/LR', self.scheduler.get_last_lr(), step)
                    self.writer.add_scalar(f'Train/Loss', loss.item(), step)
                    # self.writer.add_scalar(f'Train/Temp Loss', tempLoss.item(), step)
                    img = torchvision.utils.make_grid(orig_img)
                    self.writer.add_image(f'Train/Image', img, step)

                    if not self.args.mono_ablation:
                        left_pd = torchvision.utils.make_grid(orig_left_pd)
                        self.writer.add_image(f'Train/Left DP', left_pd, step)
                        right_pd = torchvision.utils.make_grid(orig_right_pd)
                        self.writer.add_image(f'Train/Right DP', right_pd, step)

                    uni_disp = torchvision.utils.make_grid(disp)
                    self.writer.add_image('Train/Uni Disparity', uni_disp, step)
                    scld_disp = torchvision.utils.make_grid(torch.abs(disp))
                    self.writer.add_image('Train/Scaled Disparity', scld_disp, step)
                    for j in range(min(4, self.args.batchsize)):
                        self.writer.add_video(f"Train/Predicted LF-{j}", pred_lf[j:j+1,...].clip(0, 1), step)
                        # warped_lf = logs
                        # self.writer.add_video(f"Train/Warped LF-{j}", warped_lf[j:j+1,...].clip(0, 1), step)
                        
                step += 1
                self.scheduler.step()


    def calculate_ssim(self, img1, img2):
        with torch.no_grad():
            ssim = SSIMLoss()
            N, V, C, H, W = img1.shape
            img1 = img1.reshape(N*V, C, H, W).cpu()
            img2 = img2.reshape(N*V, C, H, W).cpu()
            loss = 1 - 0.1*ssim(img1, img2)
            return loss
        

    def validate(self, val_loader, max_disp, calcMetrics = False):
        ###############################################################################################
        self.model.eval()
        
        ###############################################################################################
        # some globals
        iters = len(val_loader)
        step = self.epoch * iters
        loss_avg = RunningAverage()
        psnr_avg = RunningAverage()
        ssim_avg = RunningAverage()
        # lpips_avg = RunningAverage()
        mse_avg = RunningAverage()

        all_psnr = []
        all_ssim = []

        avg_psnr = 0.
        avg_ssim = 0.

        if calcMetrics:
            f = open(os.path.join(f'metrics/{self.args.dataset}/ref_results.txt'), 'a+')

        ################################# Validation loop #############################################
        with torch.no_grad():
            with tqdm(enumerate(val_loader), total=len(val_loader), 
                      desc='E{}/{}:V-{}, {}'.format(self.epoch+1, self.args.epochs,
                      max_disp),
                      ncols=100) as vepoch:
                for i, batch in vepoch:
                    img = batch['center'].to(self.device)
                    if "nextRGB" in batch:
                        nextRGB = batch['nextRGB'].to(self.device)
                    else:
                        nextRGB = None
                    orig_img = denormalize3d(img, self.device)


                    if not self.args.mono_ablation:
                        left_pd = batch['left_pd']
                        right_pd = batch['right_pd']

                        if self.args.rgb_only_ablation_experiment:
                            left_pd = torch.zeros_like(left_pd).to(self.device)
                            right_pd = torch.zeros_like(right_pd).to(self.device)
                        else:
                            left_pd = left_pd.to(self.device)
                            right_pd = right_pd.to(self.device)

                        orig_left_pd = denormalize1d(left_pd, self.device)
                        orig_right_pd = denormalize1d(right_pd, self.device)

                    disp = batch['disp'].to(self.device) * max_disp
                    
                    if self.args.depth_input:
                        inp = torch.cat([img, left_pd, right_pd, disp], dim=1)
                    elif self.args.mono_ablation:
                        inp = torch.cat([img, disp], dim=1)
                    else:
                        inp = torch.cat([img, left_pd, right_pd], axis=1)

                    # Pass None into dummy temporal state variable if temporal=False
                    decomposition, depth_planes, _ = self.model(inp, None)
                    if self.args.adaptive_td_ablation:
                        depth_planes = torch.Tensor([-1., 0., 1.]).to(self.device).unsqueeze(0)
                    
                    pred_lf = self.val_td_model(decomposition, depth_planes)

                    if self.args.finetune: # just to monitor the "extra" losses
                        gt_lf = batch['gt_lf'].to(self.device)
                        loss = self.compute_FT_losses(orig_img, pred_lf, gt_lf)
                        if self.args.calcMetrics:
                            # lpips_loss = self.lpips_loss(pred_lf, gt_lf)
                            # lpips_avg.append(lpips_loss.item())
                            mse_loss = self.mse_loss(pred_lf, gt_lf)
                            mse_avg.append(mse_loss.item())

                            ssim_loss = self.calculate_ssim(pred_lf, gt_lf)
                            ssim_avg.append(ssim_loss.item())
                            all_ssim.append(ssim_loss.item())

                            psnr_loss = 20 * torch.log10(1 / torch.sqrt(mse_loss))
                            all_psnr.append(psnr_loss.item())
                            psnr_avg.append(psnr_loss.item())

                            avg_psnr = sum(all_psnr) / len(all_psnr)
                            avg_ssim = sum(all_ssim) / len(all_ssim)
                            vepoch.set_postfix(av_psnr=f"{avg_psnr:0.2f}",
                                               av_ssim=f"{avg_ssim:0.2f}",
                                               in_mse=f"{mse_loss.item():0.2f}"
                                               )
                            f.write("Sample {} => PSNR: {:.2f}, SSIM: {:.2f}\n".format(i+1, psnr_loss, ssim_loss))
                            loss_avg.append(loss.item())
                    else:
                        loss, logs, temp_loss = self.compute_losses(orig_img, disp, decomposition, pred_lf, depth_planes, right=None, left=None, nextRGB=nextRGB)
                        vepoch.set_postfix(total_loss=f"{loss_avg.get_value():0.4f}({loss.item():0.4f})")
                        loss_avg.append(loss.item())
                    

                    if not self.args.dry_run and step % self.args.val_log_step == 0:
                        self.writer.add_scalar(f'Validation/Loss', loss.item(), step)
                        img = torchvision.utils.make_grid(orig_img)

                        if self.args.calcMetrics:
                            self.writer.add_scalar(f"SSIM", ssim_loss.item(), self.epoch+1)
                            # self.writer.add_scalar(f"LPIPS", lpips_loss.item(), self.epoch+1)
                            self.writer.add_scalar(f"MSE", mse_loss.item(), self.epoch+1)
                            self.writer.add_scalar(f"PSNR", psnr_loss.item(), self.epoch+1)

                        self.writer.add_image(f'Validation/Image', img, step)

                        if not self.args.mono_ablation:
                            left_pd = torchvision.utils.make_grid(orig_left_pd)
                            self.writer.add_image(f'Validation/Left DP', left_pd, step)
                            right_pd = torchvision.utils.make_grid(orig_right_pd)
                            self.writer.add_image(f'Validation/Right DP', right_pd, step)
                        
                        uni_disp = torchvision.utils.make_grid(disp)
                        self.writer.add_image('Validation/DPT Disparity', uni_disp, step)
                        scld_disp = torchvision.utils.make_grid(torch.abs(disp))
                        self.writer.add_image('Validation/Scaled Disparity', scld_disp, step)
                        
                        if not self.args.finetune: # Don't log videos for fine-tuning | Focus on metrics and fast
                            for j in range(min(4, self.args.batchsize)):
                                self.writer.add_video(f"Validation/Predicted LF-{j}", pred_lf[j:j+1,...].clip(0, 1), step)
                                # warped_lf = logs
                                # self.writer.add_video(f"Validation/Warped LF-{j}", warped_lf[j:j+1,...].clip(0, 1), step)

                    if not self.args.dry_run:
                        if loss.item() <= self.best_val_loss:
                            self.best_val_loss = loss.item()
                            # print("Saving best model...")
                            saveName = '{0:03d}.pt'.format(self.epoch)
                            model_io.save_checkpoint(self.model, self.optimizer, self.scheduler, 
                                                     self.epoch, saveName,
                                                     root=os.path.join(self.args.results, self.name, 'checkpoints'))
                            print(f"Saved best ckpt: {saveName} | best loss: {self.best_val_loss:0.4f}")
                    step += 1


                if calcMetrics:
                    f.write("--------------------\n")
                    f.write("Avg PSNR: {:.2f}, Avg SSIM: {:.2f}\n".format(avg_psnr, avg_ssim))
                    f.close()

                    # self.scheduler.step(loss_avg.get_value())


    def save_lf(self, lf, path):
        toSave = np.uint8(lf*255)
        np.save(path, toSave)


    def test(self, test_loader, max_disp):
        if self.args.save_lf:
            os.makedirs(self.args.conf_save, exist_ok=True)

        ###############################################################################################
        self.model.eval()

        ###############################################################################################
        # some globals
        iters = len(test_loader)
        # step = self.epoch * iters
        loss_avg = RunningAverage()

        ################################# Validation loop #############################################
        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_loader)):
                # print(batch)
                img = batch['center'].to(self.device)
                # orig_img = denormalize3d(img, self.device)

                # left = batch['left']#.to(self.device)
                # right = batch['right']#.to(self.device)

                
                if "seq_name" in batch:
                    seq_name = batch['seq_name'][0]
                
                if "lf_name" in batch:
                    lf_name = batch['lf_name'][0]
                    print(seq_name, lf_name)

                if not self.args.mono_ablation:
                    left_pd = batch['left_pd']
                    right_pd = batch['right_pd']
                    if self.args.rgb_only_ablation_experiment:
                        left_pd = torch.zeros_like(left_pd).to(self.device)
                        right_pd = torch.zeros_like(right_pd).to(self.device)
                    else:
                        left_pd = left_pd.to(self.device)
                        right_pd = right_pd.to(self.device)
                
                # orig_left_pd = denormalize1d(left_pd, self.device)
                # orig_right_pd = denormalize1d(right_pd, self.device)
            
                disp = batch['disp'].to(self.device) * max_disp
                if self.args.mono_ablation:
                    inp = torch.cat([img, disp], dim=1)
                elif self.args.depth_input:
                    inp = torch.cat([img, left_pd, right_pd, disp], dim=1)
                else:
                    inp = torch.cat([img, left_pd, right_pd], axis=1)
                # Pass None into dummy temporal state variable if temporal=False
                decomposition, depth_planes, _ = self.model(inp, None)

                if self.args.adaptive_td_ablation:
                    depth_planes = torch.Tensor([-1., 0., 1.]).to(self.device).unsqueeze(0)
                
                # print(depth_planes.shape, depth_planes)
                pred_lf = self.val_td_model(decomposition, depth_planes)

                # loss, logs = self.compute_losses(orig_img, disp, decomposition, pred_lf, depth_planes, right, left)
                # loss_avg.append(loss.item())
                if self.refinement:
                    lf_inp = torch.cat([pred_lf, img.unsqueeze(1)], dim=1)
                    mask, corr_lf = self.ref_model(lf_inp)
                    ref_lf = mask*corr_lf + (1-mask)*pred_lf
                    ref_lf = ref_lf.clip(0, 1)
                    pred_lf = ref_lf.cpu()

                if self.args.save_lf:
                    gt = batch['gt_lf'].cpu().numpy()[0]
                    toSave = pred_lf.detach().cpu().numpy()[0]
                    # print(toSave.shape)
                    center_toSave = toSave[self.args.angular//2, ...].transpose(1,2,0)
                    gt_toSave = gt[self.args.angular//2, ...].transpose(1,2,0)
                    # print(toSave.shape)
                    if "seq_name" in batch:
                        os.makedirs(f"{self.args.conf_save}/{self.args.dataset}/{str(seq_name)}", exist_ok=True)
                        saveStr = f"{str(seq_name)}/{str(lf_name)}"
                        saveStr2 = f"{i}_{str(seq_name)}/GT_{str(lf_name)}"
                        np.save(f"{self.args.conf_save}/{self.args.dataset}/{saveStr}", toSave, allow_pickle=True)
                        # np.save(f"{self.args.conf_save}/{self.args.dataset}/{saveStr2}", gt, allow_pickle=True)
                        # cv2.imwrite(f"{self.args.conf_save}/{self.args.dataset}/{saveStr2}_center.png", gt_toSave*255)
                        # cv2.imwrite(f"{self.args.conf_save}/{self.args.dataset}/{saveStr}_center.png", center_toSave*255)
                    else:
                        os.makedirs(f"{self.args.conf_save}/{self.args.dataset}/", exist_ok=True)
                        saveStr = f"LF_{str(i+1)}.npy"
                        saveStr2 = f"GT_LF_{str(i+1)}.npy"
                        np.save(f"{self.args.conf_save}/{self.args.dataset}/{saveStr2}", gt, allow_pickle=True)
                        np.save(f"{self.args.conf_save}/{self.args.dataset}/{saveStr}", toSave, allow_pickle=True)
                        cv2.imwrite(f"{self.args.conf_save}/{self.args.dataset}/GT_LF_{str(i+1)}_center.png", cv2.cvtColor(255*gt_toSave, cv2.COLOR_BGR2RGB))
                        cv2.imwrite(f"{self.args.conf_save}/{self.args.dataset}/LF_{str(i+1)}_center.png", cv2.cvtColor(255*center_toSave, cv2.COLOR_BGR2RGB))
                # warped_lf = logs
                # step += 1


    def fine_tune(self, train_loader):
        self.model.train()
        # for name, param in self.model.named_parameters(): # Freeze encoder 
        #     # Need to do this every-time because we are also testing after X epochs.
        #     if 'encoder' in name:
        #         param.requires_grad = False

        iters = len(train_loader)
        step = self.epoch * iters
        best_loss = np.inf
        loss_avg = RunningAverage()

        ################################# Train loop ##################################################
        torch.autograd.set_detect_anomaly(True)
        with tqdm(enumerate(train_loader), total=len(train_loader), 
                      desc='E{}/{}:T'.format(self.epoch+1, self.args.epochs),
                      ncols=100) as tepoch:
            self.writer.add_scalar("epoch", self.epoch+1, self.epoch+1)
            for i, batch in tepoch:
                max_disp = random.choice(self.disp_list)
                
                img = batch['center'].to(self.device)
                orig_img = denormalize3d(img, self.device)

                left_pd = batch['left_pd']
                right_pd = batch['right_pd']


                if self.args.rgb_only_ablation_experiment:
                    left_pd = torch.zeros_like(left_pd).to(self.device)
                    right_pd = torch.zeros_like(right_pd).to(self.device)
                else:
                    left_pd = left_pd.to(self.device)
                    right_pd = right_pd.to(self.device)
                
                # orig_left_pd = denormalize1d(left_pd, self.device)
                # orig_right_pd = denormalize1d(right_pd, self.device)
                
                # We do not have disparities for Fine-tuning datasets
                disp = batch['disp'].to(self.device) * max_disp
                
                # Dual pixel data stored after image
                # print(img.shape, left_pd.shape, right_pd.shape)
                inp = torch.cat([img, left_pd, right_pd], axis=1)
                # print(inp.shape)

                if self.args.depth_input:
                    inp = torch.cat([img, left_pd, right_pd, disp], dim=1)
                else:
                    inp = torch.cat([img, left_pd, right_pd], axis=1)
                

                # Pass None into dummy temporal state variable if temporal=False
                decomposition, depth_planes, _ = self.model(inp, None)
                pred_lf = self.train_td_model(decomposition, depth_planes)

                gt_lf = batch['gt_lf'].to(self.device)
                
                loss, logs = self.compute_losses(orig_img, disp, decomposition, pred_lf, depth_planes)
                loss_tot = loss + 10. * self.compute_FT_losses(orig_img, pred_lf, gt_lf) # 5x for more focus on MSE & LPIPS
                loss_tot = loss_tot / self.args.accumulation_steps # To account for Gradient Accumulation
                loss_tot.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                if (i+1) % self.args.accumulation_steps == 0 or (i+1) == len(train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                loss_avg.append(loss.item())

                step += 1
                # self.scheduler.step() # NOTE: No scheduling
        

    def main_worker(self):
        ###############################################################################################
        
        ##################################### Dataloaders #############################################
        if self.mode == 'train':
            train_loader = LFDataLoader(self.args, 'train').data
            val_loader = LFDataLoader(self.args, 'eval').data
        ###################################### Scheduler ##############################################
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                             max_lr=self.args.lr,
                                                             steps_per_epoch=len(train_loader),
                                                             epochs=self.args.epochs,
                                                             pct_start=0.1,)
        if self.mode == 'test':
            test_loader = LFDataLoader(self.args, 'test').data
        
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
                            print("[train.py] found latest ckpt:", latest)
    
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

            max_disp_list = [1.5]

            if args.finetune:
                print(f"Fine-tuning on {self.args.dataset}...")

                for epoch in range(self.start_epoch, self.args.epochs):
                    self.epoch = epoch
                    self.fine_tune(train_loader)
                    if epoch % self.args.validate_every == 0:
                        # max_disp_list = [1.5, 2.0]
                        for max_disp in max_disp_list:
                            self.validate(val_loader, max_disp, calcMetrics=True)

            else: # Standard continue training
                for epoch in range(self.start_epoch, self.args.epochs):
                    self.epoch = epoch
                    self.train(train_loader)
                    if epoch % self.args.validate_every == 0:
                        # max_disp_list = [1.5, 2.0]
                        for max_disp in max_disp_list:
                            self.validate(val_loader, max_disp)

            if args.test_after_full_train:
                for max_disp in max_disp_list:
                    self.test(test_loader, max_disp)


        elif self.mode == 'test':
            assert args.resume_id is not None, "Resume ID not given"
            print(f"[+] Trying to test using {args.resume_id}...")
            self.name = args.resume_id
            if os.path.isdir(os.path.join(args.results, args.resume_id)):
                # print('Logging directory: {}/{}'.format(args.results, args.resume_id))
                directory = os.path.join(args.results, args.resume_id, 'checkpoints')
                try:
                    path = os.path.join(directory, "029.pt")
                    checkpoint = torch.load(path, map_location='cpu')['model']
                    self.model = model_io.load_weights(self.model, checkpoint)
                    self.model = self.model.to(self.device)
                    print('\nTesting using ckpt at path: {}'.format(path))

                except Exception as e:
                    print("Error:", e)
                    print("\n[!] FAILED to Test! :(")
                        
                if not self.args.save_lf:
                    self.writer = SummaryWriter(os.path.join(args.results, args.resume_id))

                self.test(test_loader, self.args.baseline, 0.3)

        else:
            print("Invalid mode. Choose from train/test/eval")


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
    parser.add_argument('-exp', '--exp_name', default='pp', type=str, help='name of the experiment')
    parser.add_argument('-d', '--dry_run', default=False, action='store_true', help='choose to save checkpoints')
    parser.add_argument('--mode', default='test', type=str, help='train/test option')

    parser.add_argument('--finetune', default=False, action='store_true', help='fine-tune on a new dataset')

    parser.add_argument('--log_step', default=1000, type=int, help='gaps to log training steps')
    parser.add_argument('--val_log_step', '-vls', default=500, type=int, help='gaps to log validation steps')
    parser.add_argument('-r', '--resume', default=False, action='store_true', help='Resume from checkpoint')
    parser.add_argument('-rid', '--resume_id', default="AddLN_GA16_RGB+DP_30@21-Aug_17:59:46", type=str, help='experiment to resume')
    parser.add_argument('--validate_every', default=1, type=int, help='validate once every k epoch')
    
    parser.add_argument('--test_after_full_train', default=False, action='store_true', help='test after training')
    
    parser.add_argument('--results', default='results', type=str, help='directory to save results')
    
    parser.add_argument('--gpu', default="1", type=str, help='which gpu(s) to use')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loading')
    parser.add_argument('--rgb_only_ablation_experiment', default=False, action='store_true', help='use only rgb images for training')
    
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
    # './FT_train-test_files/Selected_Seq/'
    parser.add_argument('--filenames_file_folder',
                        default='./FT_train-test_files/Selected_Seq/',
                        type=str, help='path to the folder containing filenames to use')

    parser.add_argument('--visualization_shuffle', default=False, action='store_true', help='visualize input data')

    # parser.add_argument('--isConf', default=False, action='store_true', help='whether to save complete LF for conference')
    parser.add_argument('--conf_save', default='./ours_1.x', type=str, help='path to save conference results')
    
    ############################################# Model I/O parameters ######################################
    parser.add_argument('-th', '--train_height', type=int, help='train height', default=256)
    parser.add_argument('-tw', '--train_width', type=int, help='train width', default=192)
    parser.add_argument('-vh', '--val_height', type=int, help='validate height', default=256)
    parser.add_argument('-vw', '--val_width', type=int, help='validate width', default=192)
    parser.add_argument('--depth_input', default=False, action='store_true', 
                        help='whether to use depth as input to network')

    ####################################### Loss parameters ###########################################
    parser.add_argument('-wgt', '--edge_weight_mask', default=False, action='store_true', 
                        help='whether to use edge weights mask')
    parser.add_argument('-ss', '--ssim', default=False, action='store_true',
                        help='whether to use SSIM loss')
    parser.add_argument('-ws', '--w_ssim', default=0.5, type=float, help='weight value for SSIM loss')
    parser.add_argument('-lc', '--lf_consistency', default=1.0, type=float, 
                        help='weight value for geometric consistency')
    parser.add_argument('-tc', '--temp_consistency', default=0.5, type=float, 
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

    ####################################### CVT parameters ############################################
    parser.add_argument('-dpca', '--dp_cross_attention', default=False, action='store_true', 
                    help='Switch on cross attention')
    
    ##################################### Learning parameters #########################################
    parser.add_argument('-e', '--epochs', default=30, type=int, help='number of total epochs to run')
    parser.add_argument('-bs', '--batchsize', default=1, type=int, help='batch size')
    parser.add_argument('-lr', '--lr', default=5e-4, type=float, help='max learning rate')
    parser.add_argument('-wd', '--wd', default=5e-4, type=float, help='weight decay')
    parser.add_argument('-acc', '--accumulation_steps', default=16, type=int, help='gradient accumulation steps')
    
    ##################################### Tensor Display parameters #########################################
    parser.add_argument('-m', '--rank', default=12, type=int, help='rank of the tensor display')
    parser.add_argument('-l', '--num_layers', default=3, type=int, help='number of layers in the tensor display')
    parser.add_argument('-v', '--angular', default=7, type=int, help='number of angular views to output')
    parser.add_argument('-tdf', '--td_factor', default=1, type=int, help='disparity factor for layers')

    #### Finetuning & Metrics
    parser.add_argument('--dataset', default='Selected_Seq', type=str, help='Dataset to train on')
    parser.add_argument('--genDP_path', default='/data2/aryan/Stanford/test_dp', type=str, help='path to generated dual pixels') 
    parser.add_argument('--calcMetrics', help='if set, calculates metrics', action='store_true', default=False)
    parser.add_argument('--lf_path', default='/media/data/prasan/datasets/LF_video_datasets/', type=str, help='path to light field dataset')
    parser.add_argument('-ty', '--type', type=str, default='resize', 
                        help='whether to train with crops or resized images')
    parser.add_argument('-cc', '--color_corr', default=True, action='store_true', help='color correct images')
    parser.add_argument('--otherDS_disp_path', default='/data/prasan/datasets/LF_video_datasets/DPT-depth', type=str, help='path to other datasets disparity maps')
    parser.add_argument('--model_summary', default=False, action='store_true', help='print model summary')
    parser.add_argument('--save_lf', default=False, action='store_true', help='save light field')
    
    parser.add_argument('--adaptive_td_ablation', default=False, action='store_true', help='do not use adaptive td')
    parser.add_argument('--refinement', default=False, action='store_true')

    parser.add_argument('-lab', '--loss_ablation', default="bruh", type=str, help='loss ablation version(see table-2/ablation-1 inpaper)')
    parser.add_argument('--mono_ablation', default=False, action='store_true', help='use RGB+Disp for training')
    parser.add_argument('--ft_temp', default=False, action='store_true', help='use temporal loss')
    parser.add_argument('--baseline', default=1.5, type=float, help='baseline for testing')
    parser.add_argument('--refinement', default=False, action='store_true', help='use refinement')
    args = parser.parse_args()

    if args.results != '.' and not os.path.isdir(args.results):
        os.makedirs(args.results)

    trainer = Trainer(args)
    trainer.main_worker()
    # wandb.finish()
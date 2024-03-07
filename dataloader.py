import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from einops import rearrange
from datetime import datetime
import pandas as pd
import glob
import cv2


def preprocessing_transforms(mode, size):
    return transforms.Compose([ToTensor(mode=mode, size=size)])


def preprocessing_transforms2(mode, size, get_type='resize'):
    return transforms.Compose([ToTensor2(mode=mode, size=size, get_type=get_type)])


class LFDataLoader(object):
    def __init__(self, args, mode):
        if mode == "train":
            size = (args.train_height, args.train_width)
            if not args.finetune:
                self.training_samples = DPDataset(
                    args, mode, transform=preprocessing_transforms(mode, size)
                )
                self.train_sampler = None
                self.data = DataLoader(
                    self.training_samples,
                    args.batchsize,
                    shuffle=(self.train_sampler is None),
                    num_workers=args.workers,
                    pin_memory=True,
                    sampler=self.train_sampler,
                )
            else:
                self.training_samples = FineTuneDPDataset(
                    args, mode, transform=preprocessing_transforms2(mode, size, get_type='resize')
                )
                self.train_sampler = None
                self.data = DataLoader(
                    self.training_samples,
                    args.batchsize,
                    shuffle=(self.train_sampler is None),
                    num_workers=args.workers,
                    pin_memory=False,
                    sampler=self.train_sampler,
                )
            

        elif mode == "eval":
            size = (args.val_height, args.val_width)
            if not args.finetune:
                self.testing_samples = DPDataset(
                    args, mode, transform=preprocessing_transforms(mode, size)
                )
            else: # Finetuning dataset
                self.testing_samples = FineTuneDPDataset(
                    args, mode, transform=preprocessing_transforms2(mode, 
                                                                    (args.train_height, args.train_width), 
                                                                    get_type='resize')
                )
            self.eval_sampler = None
            self.data = DataLoader(
                    self.testing_samples,
                    args.batchsize,
                    shuffle=args.visualization_shuffle,
                    num_workers=1,
                    pin_memory=True,
                    sampler=self.eval_sampler,)
            
        
        elif mode == "test":
            # print("[!] In mode test! Really cool stuff!")
            size = (args.val_height, args.val_width)
            if args.dataset == "Pixel4":
                self.testing_samples = DPDataset(
                    args, mode, transform=preprocessing_transforms(mode, size))
            else:
                self.testing_samples = FineTuneDPDataset(
                    args, mode, transform=preprocessing_transforms2(mode, 
                                                                    (args.train_height, args.train_width), 
                                                                    get_type='resize')
                )

            self.eval_sampler = None
            self.data = DataLoader(
                self.testing_samples,
                args.batchsize,
                shuffle=args.visualization_shuffle,
                num_workers=1,
                pin_memory=False,
                sampler=self.eval_sampler,
            )

        else:
            print("mode should be one of 'train or eval'. Got {}".format(mode))


class FineTuneDPDataset(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        if mode == 'test':
            with open(os.path.join("FT_train-test_files", args.dataset, "test_files.txt"), 'r') as f:
                self.filenames = f.readlines()
        elif mode == 'train':
            with open(os.path.join("FT_train-test_files", args.dataset, "train_files.txt"), 'r') as f:
                self.filenames = f.readlines()
        elif mode == 'eval':
            with open(os.path.join("FT_train-test_files", args.dataset, "eval_files.txt"), 'r') as f:
                self.filenames = f.readlines()

        self.type = args.type
        self.mode = mode
        self.transform = transform
        self.color_corr = args.color_corr

        # Fine tune at higher resolution
        self.width = args.val_width
        self.height = args.val_height

    def __len__(self):
        return len(self.filenames)

    def color_correct(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def toGray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def CWH2HWC(self, img):
        return np.transpose(img, (1, 2, 0))

    def elementwise_avg(self, img, img_X):
        h,w = img.shape
        dp_img = np.zeros((h,w), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                dp_img[i,j] = (1. * img[i,j] + img_X[i,j]) / 2.
        return dp_img

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        paths = sample_path.split('\t')
        # paths[0] -> LF path
        # paths[1] -> left dp path
        # paths[2] -> right dp path
        # paths[3] -> disp path
        sample = {}

        lf_path = os.path.join(self.args.lf_path, paths[0])
        lf = np.load(lf_path) / 255.0
        if lf.shape[4] == 3:
            lf = lf.transpose([0, 1, 4, 2, 3])
        else:
            lf = lf.transpose([1, 0, 2, 3, 4])
        if self.color_corr:
            mean = lf.mean()
            fact = np.log(0.4) / np.log(mean)
            if fact<1:
                lf = lf ** fact

        X, Y, C, H, W = lf.shape
        image = lf[X//2, Y//2, ...] # Center view

        ### TEMP
        img = np.array(image.copy()).astype('float32')
        img_left = np.array(lf[X//2, (Y//2) - 1, ...]).astype('float32')
        img_right = np.array(lf[X//2, (Y//2) + 1, ...]).astype('float32')

        # print(img_left.shape, img_right.shape)
        img, img_left, img_right = self.CWH2HWC(img), self.CWH2HWC(img_left), self.CWH2HWC(img_right)
        img, img_left, img_right = self.toGray(self.color_correct(img)), self.toGray(self.color_correct(img_left)), self.toGray(self.color_correct(img_right))

        left_dp_channel = self.elementwise_avg(img, img_left) 
        right_dp_channel = self.elementwise_avg(img, img_right)

        right_dp = np.expand_dims(right_dp_channel, axis=0)
        left_dp = np.expand_dims(left_dp_channel, axis=0)

        # print(left_dp_channel.shape)
        ### TEMP END
        # print("image.shape: ", image.shape)
        lf = lf.reshape(X*Y, C, H, W)
        
        disp_path = os.path.join(self.args.otherDS_disp_path, paths[0].split(".")[0]+".png")
        # print(os.path.join(self.args.otherDS_disp_path, paths[0]))
        disp = cv2.imread(disp_path, cv2.IMREAD_ANYDEPTH) / 255.
        disp = (disp - disp.min()) / (disp.max() - disp.min())
        disp = 1 - disp
        # print(disp.shape, disp.dtype, disp.max(), disp.min())
        # disp = cv2.resize(disp, (self.width, self.height), interpolation=cv2.INTER_CUBIC) / 255.
        disp = np.expand_dims(disp, axis=0)
        # disp = np.zeros_like(img)
        # disp = np.expand_dims(disp, axis=0)
        # [!] NOTE: Uncomment below if you want to use PRECOMPUTED left and right views as well
        # left_dp = Image.open(os.path.join("/data2/aryan/", paths[1])).resize((self.width, self.height))
        # left_dp = np.array(left_dp, dtype=np.float32) / 255.0 
        # left_dp = np.expand_dims(left_dp, axis=0)
        # # print(left_dp.shape)
        # # print(f"[!] Left_dp.shape: {left_dp.shape} | {np.amax(left_dp)} | {np.amin(left_dp)}")
        # right_dp = Image.open(os.path.join("/data2/aryan/", paths[2])).resize((self.width, self.height))
        # right_dp = np.array(right_dp, dtype=np.float32) / 255.0
        # right_dp = np.expand_dims(right_dp, axis=0)
        
        sample = {'center': image, 
                  'gt_lf': lf,
                  'left_pd': left_dp, 
                  'right_pd': right_dp,
                  'disp': disp, 
                  'seq_name': paths[0].split('/')[-2],
                  'lf_name': paths[0].split('/')[-1],
                  }

        if self.transform:
            sample = self.transform(sample)

        return sample


class DPDataset(Dataset):
    def __init__(self, args, mode, transform=None):
        self.datapath = args.datapath  # Path to rectified dataset
        self.unrect_datapath = args.unrect_datapath  # Path to unrectified dataset

        self.use_unimatch = args.use_unimatch
        if self.use_unimatch:
            self.unimatch_disp_path = args.unimatch_disp_path
        else:
            self.unimatch_disp_path = None

        self.mode = mode

        if mode == "eval":
            with open(os.path.join(args.filenames_file_folder, 'val_files.txt'), "r") as f:
                self.filenames = f.readlines()
            print("Validation dataset length: ", len(self.filenames))
            self.height = args.val_height 
            self.width = args.val_width 

        elif mode == "train":
            with open(os.path.join(args.filenames_file_folder, 'train_files.txt'), "r") as f:
                self.filenames = f.readlines()
            print("Train dataset length: ", len(self.filenames))
            self.height = args.train_height
            self.width = args.train_width

        elif mode == 'test':
            with open(os.path.join(args.filenames_file_folder, 'test_files.txt'), "r") as f:
                self.filenames = f.readlines()
            print("Test dataset length: ", len(self.filenames))
            self.height = args.val_height
            self.width = args.val_width
        else:
            print("Invalid Mode Specified. Must be one of 'train' or 'eval' or 'test'")
            exit()

        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    # Convert to getting a single frame at a time
    def __getitem__(self, index):
        rgb_file, dp_left_file, dp_right_file, dpt_file = self.filenames[index][:-1].split(';')
        
        center_img = np.array(Image.open(os.path.join(
            self.datapath, "B", "Video_data", rgb_file
        )).resize((self.width, self.height)), dtype=np.float32) / 255.
        # The one used to calculate disp maps with + used for stereoscopic photometric loss
        # check if left img file exists first
        # if os.path.exists(os.path.join(self.datapath, "C", "Video_data", rgb_file)):
        #     left_img = np.array(Image.open(os.path.join(
        #         self.datapath, "C", "Video_data", rgb_file
        #     )).resize((self.width, self.height)), dtype=np.float32) / 255.
        #     # print(f"Found left: {rgb_file}")
        # else:
        #     # print(f"\n[!]Left NA: {rgb_file}")
        #     left_img = None
        #     # print(f"[!] Left image not found for {rgb_file} in dataset")
        # if os.path.exists(os.path.join(self.datapath, "A", "Video_data", rgb_file)):
        #     right_img = np.array(Image.open(os.path.join(
        #         self.datapath, "A", "Video_data", rgb_file
        #     )).resize((self.width, self.height)), dtype=np.float32) / 255.
        #     # print(f"Found right: {rgb_file}")
        # else:
        #     # print(f"\n[!] Right NA: {rgb_file}")
        #     right_img = None
        #     # print(f"[!] Right image not found for {rgb_file} in dataset")
        # 20230320_171226_844/image_350.jpg
        num = rgb_file.split("_")[-1].split(".")[0]
        next_rgb_file = rgb_file.replace(num, str(int(num)-1).zfill(3))

        if os.path.exists(os.path.join(self.datapath, "B", "Video_data", next_rgb_file)):
            nextRGB = np.array(Image.open(os.path.join(
                self.datapath, "B", "Video_data", next_rgb_file
            )).resize((self.width, self.height)), dtype=np.float32) / 255.
        else:
            nextRGB = np.zeros_like(center_img)

        left_pd_img = np.array(Image.open(os.path.join(
            self.unrect_datapath, "B", "dp_data", dp_left_file
        )).rotate(
            270, expand=True
        ).resize((self.width, self.height)), dtype=np.float32)[:,:,0:1] 
        right_pd_img = np.array(Image.open(os.path.join(
            self.unrect_datapath, "B", "dp_data", dp_right_file
        )).rotate(
            270, expand=True
        ).resize((self.width, self.height)), dtype=np.float32)[:,:,0:1]
        # Undo tone map and normalize 16 bit DP data
        # left_pd_img = (left_pd_img)**2 / ((2**16-1)*1.0)
        # right_pd_img = (right_pd_img)**2 / ((2**16-1)*1.0)
        # Normalize 8 bit DP data
        left_pd_img = left_pd_img / 255.
        right_pd_img = right_pd_img / 255.
        
        if self.use_unimatch:
            # get video name
            disp = cv2.imread(os.path.join(self.unimatch_disp_path, dpt_file.split('.')[0] + '_disp.png'), 
                              cv2.IMREAD_ANYDEPTH)
            disp = cv2.resize(disp, (self.width, self.height), interpolation=cv2.INTER_CUBIC) / 255.
        else:
            disp = np.array(Image.open(os.path.join(
                self.datapath, "B", "dptdepth", dpt_file
            )).resize((self.width, self.height)), dtype=np.float32) / 255.
        
        disp = disp[..., None] # Add channel dimension
        
        sample = {'center': center_img, # 'left': left_img, 'right': right_img,
                  'left_pd': left_pd_img, 'right_pd': right_pd_img,# 'left_disp': left_disp, 'right_disp': right_disp
                  'disp': disp, 'nextRGB': nextRGB}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    def __init__(self, mode, size):
        self.mode = mode
        self.H, self.W = size
        self.normalize_3d = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize_1d = transforms.Normalize(mean=[0.456], std=[0.224])
        # self.transform = transforms.Resize(size)


    def __call__(self, sample):
        trans_sample = {}
        for key in ['center', 'nextRGB']:#, 'left', 'right']:
            if sample[key] is None:
                trans_sample[key] = np.array([-3407])
                continue
            image = sample[key]
            image = self.to_tensor(image)
            # image = self.transform(image)
            image = self.normalize_3d(image)
            trans_sample[key] = image
        
        for key in ['left_pd', 'right_pd']:
            image = sample[key]
            image = self.to_tensor(image)
            # image = self.transform(image)
            image = self.normalize_1d(image)
            trans_sample[key] = image

        for key in ['disp']:#['left_disp', 'right_disp']:
            image = sample[key]
            #print(image.max(), image.min())
            image = self.to_tensor(image)
            #_, H, W = image.shape
            #image = image * self.W / W
            # image = self.transform(image)
            trans_sample[key] = image


        return trans_sample


    def to_tensor(self, pic):
        image = torch.FloatTensor(pic)
        shape = image.shape
        if len(shape) == 3:
            image = image.permute(2, 0, 1)
        elif len(shape) == 4:
            image = image.permute(0, 3, 1, 2)

        return image

    
class ToTensor2(object):
    def __init__(self, mode, size, get_type):
        self.mode = mode
        self.normalize_3d = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize_1d = transforms.Normalize(mean=[0.456], std=[0.224])
        self.transform = transforms.Resize(size)
        self.type = get_type


    def to_tensor(self, pic):
        image = torch.FloatTensor(pic)
        
        shape = image.shape
        if len(shape) == 3 and shape[-1] == 3:
            image = image.permute(2, 0, 1)
        elif len(shape) == 4 and shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        
        return image


    def __call__(self, sample):
        image = sample['center']
        image = self.to_tensor(image)
        if self.type == 'resize':
            image = self.transform(image)
        image = self.normalize_3d(image)

        lf = sample['gt_lf']
        lf = self.to_tensor(lf)
        if self.type == 'resize':
            lf = self.transform(lf)
            
    
        left_dp = sample['left_pd']
        left_dp = self.to_tensor(left_dp)
        if self.type == 'resize':
            left_dp = self.transform(left_dp)
        left_dp = self.normalize_1d(left_dp)


        right_dp = sample['right_pd']
        right_dp = self.to_tensor(right_dp)
        if self.type == 'resize':
            right_dp = self.transform(right_dp)
        right_dp = self.normalize_1d(right_dp)

        disp = sample['disp']
        disp = self.to_tensor(disp)
        if self.type == 'resize':
            disp = self.transform(disp)
        

        sample = {'center': image, 'gt_lf': lf, 'left_pd': left_dp, 'right_pd': right_dp, 'disp': disp, 
                  'seq_name': sample['seq_name'], 'lf_name': sample['lf_name']}
            
        return sample


if __name__ == "__main__":

    class Args:
        def __init__(self):
            # self.filenames_file = "train_inputs/train_skip5.json" # OLD STYLE, will fail
            self.filenames_file_folder = "./train_inputs/Pixel4_3DP_frame_skip3"
            self.datapath = "/data2/raghav/datasets/Pixel4_3DP/rectified"
            self.unrect_datapath = "/data2/raghav/datasets/Pixel4_3DP/unrectified"
            self.train_height = 600
            self.train_width = 800
    args = Args()
    dataset = DPDataset(args, "train", transform=preprocessing_transforms("train", (args.train_height, args.train_width)))
    dLoader = DataLoader(dataset=dataset, batch_size=2)
    # print(len(dataset))
    # print(len(dLoader))
    
    for i, data in enumerate(dLoader):
        print(data["disp"].shape)
        break
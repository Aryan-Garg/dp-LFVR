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
import cv2
import glob


def preprocessing_transforms(mode, size):
    return transforms.Compose([ToTensor(mode=mode, size=size)])


def preprocessing_transforms2(mode, size, get_type='resize'):
    return transforms.Compose([ToTensor2(mode=mode, size=size, get_type=get_type)])


class LFDataLoader(object):
    def __init__(self, args, mode, calcMetrics=False):
        if mode == "train":
            size = (args.train_height, args.train_width)
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

        elif mode == "eval":
            size = (args.val_height, args.val_width)
            self.testing_samples = DPDataset(
                args, mode, transform=preprocessing_transforms(mode, size)
            )
            self.eval_sampler = None
            self.data = DataLoader(
                self.testing_samples,
                args.batchsize,
                shuffle=args.visualization_shuffle,
                num_workers=1,
                pin_memory=True,
                sampler=self.eval_sampler,
            )

        elif mode == 'test':
            size = (args.val_height, args.val_width)
            if calcMetrics:
                self.testing_samples = DataLoadPreprocess(args, mode, 
                                                          transform=preprocessing_transforms2(mode, size))
                self.data = DataLoader(self.testing_samples, args.batchsize,
                               shuffle=False, num_workers=1,
                               pin_memory=True, sampler=None)
            else:
                self.testing_samples = DPDataset(args, mode, transform=preprocessing_transforms(mode, size))
                self.eval_sampler = None

                self.data = DataLoader(
                    self.testing_samples,
                    args.batchsize,
                    shuffle=args.visualization_shuffle,
                    num_workers=1,
                    pin_memory=True,
                    sampler=self.eval_sampler,
                )

        else:
            print("mode should be one of 'train or eval or test'. Got {}".format(mode))


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        if mode == 'test':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.type = args.type
        self.genDP_path = args.genDP_path
        self.mode = mode
        self.transform = transform
        self.color_corr = args.color_corr
        self.width = args.val_width
        self.height = args.val_height


    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        paths = sample_path.split('\t')[:-1]
        sample = {}

        for i, path in enumerate(paths):
            lf_num = path.split('/')[-1].split('-')[1].split('.')[0]
            lf_path = os.path.join(self.args.lf_path, path)
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
            # print("image.shape: ", image.shape)
            lf = lf.reshape(X*Y, C, H, W)
            
            if "unimatch" in self.args.otherDS_disp_path:
                disp_path = os.path.join(self.args.otherDS_disp_path, 
                                     self.args.dataset, 
                                     "test", 
                                     f"left_lf-{lf_num}_disp.png")
                disp = cv2.imread(disp_path, cv2.IMREAD_ANYDEPTH)
                disp = cv2.resize(disp, (self.width, self.height), interpolation=cv2.INTER_CUBIC) / 255.
            elif "DPT" in self.args.otherDS_disp_path: # dpt 
                disp_path = os.path.join(self.args.otherDS_disp_path, self.args.dataset,
                                         "test", f"lf-{lf_num}.png")
                disp = cv2.imread(disp_path, cv2.IMREAD_ANYDEPTH)
                disp = cv2.resize(disp, (self.width, self.height), interpolation=cv2.INTER_CUBIC) / 255.
            else:
                disp = np.zeros((self.height, self.width))
                
            disp = np.expand_dims(disp, axis=0)
            
            left_dp = Image.open(
                os.path.join(self.genDP_path, f"left_lf-{lf_num}.png")
                ).resize((self.width, self.height))
            left_dp = np.array(left_dp, dtype=np.float32) / 255.0 
            left_dp = np.expand_dims(left_dp, axis=0)
            # print(f"[!] Left_dp.shape: {left_dp.shape} | {np.amax(left_dp)} | {np.amin(left_dp)}")

            right_dp = Image.open(
                os.path.join(self.genDP_path, f"right_lf-{lf_num}.png")
                ).resize((self.width, self.height))
            right_dp = np.array(right_dp, dtype=np.float32) / 255.0
            right_dp = np.expand_dims(right_dp, axis=0)
            
            sample[i] = {'rgb': image, 
                         'lf': lf,
                         'disp': disp, 
                        'left_dp': left_dp, 
                        'right_dp': right_dp}

        if self.transform:
            sample = self.transform(sample)

        return sample


    def __len__(self):
        return len(self.filenames)
    

class DPDataset(Dataset):
    """
    PyTorch dataset class for the dual pixel video dataset
    """

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
            with open(os.path.join(args.filenames_file_folder, 'val_files.json'), "r") as f:
                self.metadata = json.loads(f.read())
            self.height = args.val_height
            self.width = args.val_width
        elif mode == "train":
            with open(os.path.join(args.filenames_file_folder, 'train_files.json'), "r") as f:
                self.metadata = json.loads(f.read())
            print(len(self.metadata['videos']))
            self.height = args.train_height
            self.width = args.train_width
        elif mode == 'test':
            with open(os.path.join(args.filenames_file_folder, 'test_files.json'), "r") as f:
                self.metadata = json.loads(f.read())
            print(len(self.metadata['videos']))
            self.height = args.val_height
            self.width  = args.val_width
        else:
            print(f"[!] Incorrect mode: {mode} passed to DPDataset")

        for video in self.metadata["videos"]:
            if len(video["frames"]) != self.metadata["video_length"]:
                raise Exception(
                    f"All videos need to be of the length specified in the metadata. Given video {video['video_name']} is of length {len(video['frames'])}, while expected is {self.metadata['video_length']} "
                )

        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata["videos"])

    def __getitem__(self, index):
        video = self.metadata["videos"][index]
        # times = {
        #     'initial': [],
        #     'rgb': [],
        #     'dp': [],
        #     'dpt': [],
        # }        
        # TODO: Modify to take in account multiple views (A,B,C). Current implementation just loads B view
        video_data = []

        for frame in video["frames"]:
            frame_data = {}
            # times['initial'].append(datetime.now().timestamp())
            orig_rgb_frame = np.array(Image.open(
                os.path.join(
                    self.datapath,
                    "B",
                    self.metadata["rgb_path"],
                    video["name"],
                    frame["rgb"]["B"],
                )
            ).resize((self.width, self.height)), dtype=np.float32) / 255.

            # times['rgb'].append(datetime.now().timestamp())
            
            left_dp = Image.open(
                os.path.join(
                    self.unrect_datapath,
                    "B",
                    self.metadata["dp_path"],
                    video["name"],
                    frame["left_dp"]["B"],
                )
            ).resize((self.width, self.height))
            left_dp = np.array(left_dp, dtype=np.float32)[:,:,0:1] / 255.0 # Extract only one channel as image is grayscale

            right_dp = Image.open(
                os.path.join(
                    self.unrect_datapath,
                    "B",
                    self.metadata["dp_path"],
                    video["name"],
                    frame["right_dp"]["B"],
                )
            ).resize((self.width, self.height))
            right_dp = np.array(right_dp, dtype=np.float32) [:,:,0:1]/ 255.0
            
            # times['dp'].append(datetime.now().timestamp())

            dpt_file = frame["dptdepth"]["B"]
            if self.use_unimatch:
                # get video name
                # print("[@151][vid_dataloader]: ", os.path.join(self.unimatch_disp_path, video["name"], dpt_file.split('.')[0] + '_disp.png'))    
                disp = cv2.imread(os.path.join(self.unimatch_disp_path, video["name"], dpt_file.split('.')[0] + '_disp.png'), 
                                  cv2.IMREAD_ANYDEPTH)
                disp = cv2.resize(disp, (self.width, self.height), interpolation=cv2.INTER_CUBIC) / 255.
            
            else:
                disp = np.array(Image.open(os.path.join(
                    self.datapath, "B", "dptdepth", dpt_file)).resize((self.width, self.height)), dtype=np.float32) / 255.

            disp = disp[..., None] # Add channel dimension

            # times['dpt'].append(datetime.now().timestamp())
            
            frame_data["rgb"] = orig_rgb_frame
            frame_data["left_dp"] = left_dp
            frame_data["right_dp"] = right_dp
            frame_data["disp"] = disp

            if self.transform:
                frame_data = self.transform(frame_data)

            video_data.append(frame_data)

        # times = pd.DataFrame(times)
        # times['dpt'] = times['dpt'] - times['dp']
        # times['dp'] = times['dp'] - times['rgb']
        # times['rgb'] = times['rgb'] - times['initial']
        # print(times)

        return video_data


class ToTensor(object):
    def __init__(self, mode, size):
        self.mode = mode
        self.H, self.W = size
        self.normalize_3d = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize_1d = transforms.Normalize(mean=[0.456], std=[0.224])
        # self.transform = transforms.Resize(size)


    def __call__(self, sample):
        trans_sample = {}
        # print(sample)
        for key in ['rgb']:#, 'left_stereo', 'right_stereo']:
            image = sample[key]
            image = self.to_tensor(image)
            # image = self.transform(image)
            image = self.normalize_3d(image)
            trans_sample[key] = image
        
        for key in ['left_dp', 'right_dp']:
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
        for i in sample.keys():
            image = sample[i]['rgb']
            image = self.to_tensor(image)
            if self.type == 'resize':
                image = self.transform(image)
            image = self.normalize_3d(image)

            lf = sample[i]['lf']
            lf = self.to_tensor(lf)
            if self.type == 'resize':
                lf = self.transform(lf)
            
            # print("In ToTensor2: ")
            left_dp = sample[i]['left_dp']
            left_dp = self.to_tensor(left_dp)
            if self.type == 'resize':
                left_dp = self.transform(left_dp)
            left_dp = self.normalize_1d(left_dp)


            right_dp = sample[i]['right_dp']
            right_dp = self.to_tensor(right_dp)
            if self.type == 'resize':
                right_dp = self.transform(right_dp)
            right_dp = self.normalize_1d(right_dp)

            disp = sample[i]['disp']
            disp = self.to_tensor(disp)
            if self.type == 'resize':
                disp = self.transform(disp)

            sample[i] = {'rgb': image, 'lf': lf, 'disp': disp, 'left_dp': left_dp, 'right_dp': right_dp}

            # sample[i] = {'rgb': image, 'lf': lf, 'left_dp': left_dp, 'right_dp': right_dp}
            
        return sample


if __name__ == "__main__":

    class Args:
        def __init__(self):
            # self.filenames_file = "train_inputs/train_skip5.json" # OLD STYLE, will fail
            self.filenames_file_folder = "./train_inputs/Pixel4_3DP_skip10"
            self.datapath = "/data2/raghav/datasets/Pixel4_3DP/rectified"
            self.unrect_datapath = "/data2/raghav/datasets/Pixel4_3DP/unrectified"
            self.train_height = 600
            self.train_width = 800

    dataset = DPDataset(Args(), "train", transform=preprocessing_transforms("train"))
    dLoader = DataLoader(dataset=dataset, batch_size=2)
    print(len(dataset))
    print(len(dLoader))
    
    for i, data in enumerate(dLoader):
        print(data[7]["disp"].shape)
        break
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence

import softsplat_new as softsplat


class L1Loss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(L1Loss, self).__init__()
        self.name = 'L1 Loss'
        

    def forward(self, inp, tar, weight_mask=None, valid_mask=None):
        mask = (tar > 0).to(bool) * (inp > 0).to(bool)
        if valid_mask is not None:
            mask = mask * valid_mask.to(bool)
        minp = inp[mask]
        mtar = tar[mask]
        diff = torch.abs(minp - mtar)
        if weight_mask is not None:
            mweight = weight_mask[mask]
            diff = diff * mweight
        loss = diff.mean()
        return 10 * loss



class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super(BinsChamferLoss, self).__init__()
        self.name = "ChamferLoss"


    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss



class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2


    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return 10 * torch.mean(torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1))


class SmoothLoss(nn.Module):
    def __init__(self, args, device):
        super(SmoothLoss, self).__init__()
        self.name = 'Smoothness Loss'
        self.args = args
        gradx = torch.FloatTensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]]).to(device)
        grady = torch.FloatTensor([[-1, -2, -1],
                                   [0,   0,  2],
                                   [1,   0,  1]]).to(device)
        self.disp_gradx = gradx.unsqueeze(0).unsqueeze(0)
        self.disp_grady = grady.unsqueeze(0).unsqueeze(0)
        self.img_gradx = self.disp_gradx.repeat(1, 3, 1, 1)
        self.img_grady = self.disp_grady.repeat(1, 3, 1, 1)


    def get_smooth_loss(self, disp, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """

        grad_disp_x = torch.abs(F.conv2d(disp, self.disp_gradx, padding=1, stride=1))
        grad_disp_y = torch.abs(F.conv2d(disp, self.disp_grady, padding=1, stride=1))

        grad_img_x = torch.abs(torch.mean(F.conv2d(img, self.img_gradx, padding=1, stride=1), dim=1, keepdim=True))
        grad_img_y = torch.abs(torch.mean(F.conv2d(img, self.img_grady, padding=1, stride=1), dim=1, keepdim=True))

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        loss_x = 10 * (torch.sqrt(torch.var(grad_disp_x) + 0.15 * torch.pow(torch.mean(grad_disp_x), 2)))
        loss_y = 10 * (torch.sqrt(torch.var(grad_disp_y) + 0.15 * torch.pow(torch.mean(grad_disp_y), 2)))

        return loss_x + loss_y


    def forward(self, decomposition, disp):
        N, layers, rank, C, H, W = decomposition.shape
        disp = disp.unsqueeze(dim=1)
        disp = disp.repeat(1, layers*rank, 1, 1, 1)
        disp = disp.reshape(-1, 1, H, W)
        decomposition = decomposition.reshape(-1, C, H, W)
        loss = self.get_smooth_loss(disp, decomposition)
        return loss


class PhotometricConsistency(nn.Module):
    def __init__(self, args, device):
        super(PhotometricConsistency, self).__init__()
        self.name = 'Photometric Consistency Loss'
        self.args = args
        self.angular = args.angular
        self.device = device
        self.diff_loss = nn.L1Loss()
        self.is_ssim = args.ssim
        self.w_ssim = args.w_ssim
        self.ssim_loss = SSIMLoss().to(device)


    def forward(self, left_img, right_img, pred_lf):
        left_pred = pred_lf[:, int(self.angular**2//2)-int(self.angular//2), ...]
        right_pred = pred_lf[:, int(self.angular**2//2)+int(self.angular//2), ...]
        photo_loss = self.diff_loss(left_pred, left_img) + self.diff_loss(right_pred, right_img)
        if self.is_ssim:
            photo_loss +=  self.w_ssim * self.ssim_loss(left_pred, Left_img)
            photo_loss +=  self.w_ssim * self.ssim_loss(right_pred, right_img)
        return photo_loss



class GeometricConsistency(nn.Module):
    def __init__(self, args, device):
        super(GeometricConsistency, self).__init__()
        self.name = 'Geometric Consistency Loss'
        self.args = args
        self.angular = args.angular
        self.device = device
        self.diff_loss = L1Loss()
        self.use_mask = args.edge_weight_mask
        self.is_ssim = args.ssim
        self.w_ssim = args.w_ssim
        self.ssim_loss = SSIMLoss().to(device)
        
        lx_factor = np.arange(0, self.angular)
        rx_factor = np.arange(self.angular, 0, -1)
        y_factor = np.arange(-1 * self.angular//2 + 1, self.angular//2 + 1)
        
        lfactor = np.stack([np.meshgrid(lx_factor, y_factor)], 2)
        lfactor = lfactor.squeeze().transpose(1, 2, 0).reshape(1, self.angular**2, 2, 1, 1)
        self.lfactor = torch.FloatTensor(lfactor).to(self.device)
        
        rfactor = np.stack([np.meshgrid(rx_factor, y_factor)], 2)
        rfactor = rfactor.squeeze().transpose(1, 2, 0).reshape(1, self.angular**2, 2, 1, 1)
        self.rfactor = torch.FloatTensor(rfactor).to(self.device)

        sobel_x = np.array([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]])
        sobel_y = sobel_x.T
        self.gradx = torch.FloatTensor(sobel_x[None, None, ...]).to(device)
        self.grady = torch.FloatTensor(sobel_y[None, None, ...]).to(device)


    def gradxy(self, tensor):
        tensor = tensor.mean(dim=1, keepdim=True)
        gradx = F.conv2d(tensor, self.gradx, padding=1, stride=1)
        grady = F.conv2d(tensor, self.grady, padding=1, stride=1)
        grad = gradx.abs() + grady.abs()
        return grad


    def left_forward_warp(self, img, depth):
        N, _, H, W = depth.shape
        _, C, _, _ = img.shape

        img = img.unsqueeze(1).repeat(1, self.angular**2, 1, 1, 1)
        depth = depth.unsqueeze(1).repeat(1, self.angular**2, 2, 1, 1)
        depth = depth*self.lfactor

        depth = depth.reshape(N * self.angular**2, 2, H, W)
        img = img.reshape(N * self.angular**2, C, H, W)
        
        warped_lf = softsplat.FunctionSoftsplat(tenInput=img, tenFlow=depth, tenMetric=None, strType='average').to(self.device)
        warped_lf = warped_lf.reshape(N, self.angular**2, C, H, W)
        return warped_lf


    def right_forward_warp(self, img, depth):
        N, _, H, W = depth.shape
        _, C, _, _ = img.shape

        img = img.unsqueeze(1).repeat(1, self.angular**2, 1, 1, 1)
        depth = depth.unsqueeze(1).repeat(1, self.angular**2, 2, 1, 1)
        depth = depth*self.rfactor

        depth = depth.reshape(N * self.angular**2, 2, H, W)
        img = img.reshape(N * self.angular**2, C, H, W)
        
        warped_lf = softsplat.FunctionSoftsplat(tenInput=img, tenFlow=depth, tenMetric=None, strType='average').to(self.device)
        warped_lf = warped_lf.reshape(N, self.angular**2, C, H, W)
        return warped_lf


    def init_coord(self, img):
        _, _, h, w = img.shape
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xv, yv = np.meshgrid(x, y)
        coord = np.zeros((2, h, w))
        coord[0, ...] = xv
        coord[1, ...] = yv
        coord = np.tile(coord, [1, self.angular**2, 1, 1, 1])
        self.coord = torch.FloatTensor(coord).to(self.device)


    def get_loss(self, img, disp, factor, pred_lf):
        N, _, H, W = disp.shape
        _, V, C, _, _ = pred_lf.shape

        disp = disp.unsqueeze(1).repeat(1, self.angular**2, 2, 1, 1)
        disp = disp * factor
        disp[:, :, 0, :, :] /= W/2
        disp[:, :, 1, :, :] /= H/2
        print(disp.max(), disp.min())

        warp_coord = self.coord + disp
        warp_coord = warp_coord.reshape(N * self.angular**2, 2, H, W).permute(0, 2, 3, 1)
        pred_lf = pred_lf.reshape(N * self.angular**2, C, H, W)

        warped_lf = F.grid_sample(pred_lf, warp_coord, padding_mode='border', mode='bilinear', align_corners=True)
        warped_lf = warped_lf.reshape(N, self.angular**2, C, H, W)
        if self.use_mask:
            weight_mask = self.gradxy(img).unsqueeze(1).repeat(1, self.angular**2, 3, 1, 1)
            geo_loss = self.diff_loss(warped_lf, img.unsqueeze(1).repeat(1, self.angular**2, 1, 1, 1), weight_mask)
        else:
            geo_loss = self.diff_loss(warped_lf, img.unsqueeze(1).repeat(1, self.angular**2, 1, 1, 1))
        
        warped_lf = warped_lf.reshape(N*self.angular**2, C, H, W)
        img = img.repeat(self.angular**2, 1, 1, 1)
        if self.is_ssim:
            geo_loss +=  self.w_ssim * self.ssim_loss(warped_lf, img)

        return loss


    def forward(self, left_img, right_img, left_disp, right_disp, pred_lf):
        self.init_coord(left_img)
        
        left_geo_loss = self.get_loss(left_img, left_disp, self.lfactor, pred_lf)
        right_geo_loss = self.get_loss(right_img, right_disp, self.rfactor, pred_lf)
        geo_loss = left_geo_loss + right_geo_loss

        return geo_loss#, warped_lf
    
    '''
    def forward(self, left_img, right_img, left_disp, right_disp, pred_lf):
        left_fwrd_lf = self.left_forward_warp(left_img, left_disp)
        right_fwrd_lf = self.right_forward_warp(right_img, right_disp)

        return left_fwrd_lf, right_fwrd_lf
    '''
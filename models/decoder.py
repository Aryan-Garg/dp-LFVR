import torch
import torch.nn as nn
import torch.nn.functional as F

class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, skip_input//2, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(skip_input//2),
                                  nn.LeakyReLU(), # convtranspose halves features
                                  nn.Conv2d(skip_input//2, skip_input//2, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(skip_input//2),
                                  nn.LeakyReLU(), # maintain features
                                  nn.Conv2d(skip_input//2, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU()) # move to output dimension


    def forward(self, x, concat_with):
        # x: d, concat_with: b
        # Interpolate d to size of b
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        # Concatenate d with b along feature
        f = torch.cat([up_x, concat_with], dim=1)
        # Pass through sequence of convolutions
        return self._net(f)


class DecoderCVT(nn.Module):
    def __init__(self,
                 num_classes,
                 bottleneck_features,
                 feature_depths):
        super(DecoderCVT, self).__init__()
        self.contractor = nn.Conv2d(bottleneck_features,
                                    feature_depths[-1],
                                    kernel_size=[2,2],
                                    stride=[1,1],
                                    padding=[1,1])
        self.bn_contractor = nn.BatchNorm2d(feature_depths[-1])
        self.leaky_relu = nn.LeakyReLU()

        self.up2 = UpSampleBN(skip_input=2*feature_depths[-1],
                              output_features=feature_depths[-2])
        self.up1 = UpSampleBN(skip_input=2*feature_depths[-2],
                              output_features=feature_depths[-3])
        self.up0 = UpSampleBN(skip_input=2*feature_depths[-3],
                              output_features=feature_depths[-3])
        # Upsample scale should be same as stride of first CVT stage
        self.up_x = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(feature_depths[-3], num_classes, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU(True)


    def forward(self, x, intermediates):
        # intermediates: output of each stage of CVT

        b0,b1,b2 = intermediates
        d2 = self.contractor(x)
        d2 = self.leaky_relu(self.bn_contractor(d2)) # NOTE

        d1 = self.up2(d2, b2)
        d0 = self.up1(d1, b1)
        xd = self.up0(d0, b0)

        x = self.up_x(xd)
        x = self.conv(x)

        return self.relu(x)

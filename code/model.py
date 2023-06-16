import math
import random

import torch
from torch import nn as nn
import torch.nn.functional as F

from util.logconf import logging
from util.unetPlusPlus import ResidualUNet3D


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class UNetWrapper(nn.Module):
    #kwarg is a dictionary containing all keyword arguments passed to the constructor
    def __init__(self, **kwargs):
        super().__init__()

        self.input_batchnorm = nn.BatchNorm3d(kwargs['in_channels'])
        self.unet = ResidualUNet3D(**kwargs)

    def _init_weights(self):
        init_set = {
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Linear,
        }
        for m in self.modules():
            if type(m) in init_set:
                nn.init.kaiming_normal_(
                    m.weight.data, mode='fan_out', nonlinearity='relu', a=0
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)


    def forward(self, input_batch):
        bn_output = self.input_batchnorm(input_batch)
        un_output = self.unet(bn_output)
        
        return un_output
    
            

class SegmentationAugmentation(nn.Module):
    def __init__(
            self, flip=None, offset=None, scale=None, rotate=None, noise=None
    ):
        super().__init__()

        self.flip = flip
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

    def forward(self, input_g, label_g):
        transform_t = self._build3dTransformMatrix()
        transform_t = transform_t.expand(input_g.shape[0],-1, -1)
        transform_t = transform_t.to(input_g.device, torch.float32)
        affine_t = F.affine_grid(transform_t[:,:3],
                input_g.size(), align_corners=False)
        
        augmented_input_g = F.grid_sample(input_g,
                affine_t, padding_mode='border',
                align_corners=False)
        augmented_label_g = F.grid_sample(label_g.to(torch.float32),
                affine_t, padding_mode='border',
                align_corners=False)

        if self.noise:
            noise_t = torch.randn_like(augmented_input_g)
            noise_t *= self.noise

            augmented_input_g += noise_t

        return augmented_input_g, augmented_label_g

    
    def _build3dTransformMatrix(self):
            transform_t = torch.eye(4)

            for i in range(2):
                if self.flip:
                    if random.random() > 0.5:
                        transform_t[i,i] *= -1

                if self.scale:
                    scale_float = self.scale
                    random_float = (random.random() * 2 - 1)
                    transform_t[i,i] *= 1.0 + scale_float * random_float

            if self.rotate:
                alpha = random.random() * math.pi * 2 #Takes a random angle in radians, so in the range 0 .. 2{pi}

                rotation_z = torch.tensor([ #Rotation matrix for the 2D rotation by the random angle in the first two dimensions
                    [math.cos(alpha),-math.sin(alpha), 0, 0],
                    [math.sin(alpha), math.cos(alpha), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                
                transform_t @= rotation_z #Applies the rotation to the transformation matrix using the Python matrix multiplication operator

            return transform_t
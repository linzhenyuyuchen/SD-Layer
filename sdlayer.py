import torch
import torch.nn as nn
from .macenko import *
import numpy as np
import cv2


def OD_init(input_shape=(3,3,1,1), ref_img_path="/home/SD-Layer/Code/Train/ref_all.bmp"):
    '''This function initialized the SDLayer with Stain-Matrix obtained via SVD.'''
    squeeze_percentile = 99.9
    query = cv2.imread(ref_img_path) / 255.0
    phi,a = GetWedgeMacenko(query, squeeze_percentile)
    init = phi
    return np.reshape(init,input_shape)

class SDlayer(nn.Module):
    def __init__(self):
        super(SDlayer, self).__init__()
        self.activation = nn.Tanh()
        self.conv = nn.Conv2d(3,3,kernel_size=1)
        phi = OD_init()
        self.conv.weight = torch.nn.Parameter(torch.Tensor(phi))

    def forward(self, I):
        mask  = (~(I > 0.)) * 255.0
        I = I + mask  # this image contains 255 wherever it had 0 initially
        I_OD = - torch.log10(I/255.0)
        A = self.conv(I_OD)
        A = self.activation(A)
        return A

#model = SDlayer()
#print(list(model.parameters()))

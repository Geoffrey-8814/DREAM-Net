import torch
import torch.nn as nn
from torch.autograd import Variable

from corr import CorrTorch
from DepthwiseConv import depthwiseConv, depthwiseDeconv
from functools import partial

import numpy as np

class OpticalFlowBlock(nn.Module):
    def __init__(self, od, dd, low_feat_channels=None, scaling_factor=2, padding = 1):
        super(OpticalFlowBlock, self).__init__()
        
        self.scaling_factor = scaling_factor
        
        self.corr = CorrTorch()
        self.norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        # Define all convolutional layers and other components
        self.conv0 = depthwiseConv(od, dd[0], kernel_size=3, norm_layer = self.norm_layer, activation_layer = nn.Hardswish, stride=1)
        self.conv1 = depthwiseConv(od + dd[0], dd[1]-dd[0], kernel_size=3, norm_layer = self.norm_layer, activation_layer = nn.Hardswish, stride=1)
        self.conv2 = depthwiseConv(od + dd[1], dd[2]-dd[1], kernel_size=3, norm_layer = self.norm_layer, activation_layer = nn.Hardswish, stride=1)
        
        self._predict_flow = self.predict_flow(od + dd[2])
        
        self.classes = 5
        self._classify_motion = self.classify_motion(od + dd[2])
        
        self.activation = nn.Hardswish()
        
        if low_feat_channels != None:
            self._deconv = self.deconv(2, 2, kernel_size=4, stride=scaling_factor, padding=padding) 
            self.upfeat = self.deconv(low_feat_channels, 2, kernel_size=4, stride=scaling_factor, padding=padding)
            self.upclasses = self.deconv(self.classes, 5, kernel_size=4, stride=scaling_factor, padding=padding)#TODO
            
        
    def predict_flow(self, in_planes):
        return depthwiseConv(in_planes, 2, kernel_size=3, norm_layer = self.norm_layer, activation_layer = nn.Hardswish, stride=1)
    
    def classify_motion(self, in_planes):
        return depthwiseConv(in_planes, self.classes, kernel_size=3, norm_layer = self.norm_layer, activation_layer = nn.Hardsigmoid,stride=1)
    
    def deconv(self, in_planes, out_planes, kernel_size=4, stride=2, padding=1):
        return depthwiseDeconv(in_planes, out_planes, kernel_size, 1, stride, padding=padding)
    
    def warp(self, x, flo):# https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        
        mask = torch.autograd.Variable(torch.ones(x.size()))
        if x.is_cuda:
            mask = mask.cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())
        
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask
    
            
    def forward(self, c1, c2, flow=None, feat=None, classes=None):
        if flow ==None or feat == None:
            # Compute the correlation
            corr = self.corr(c1, c2)
            corr = self.activation(corr)
            x = corr
        else:
            up_flow = self._deconv(flow)
            up_feat = self.upfeat(feat)
            up_classes = self.upclasses(classes)

            # Warp the feature map with the flow
            warp = self.warp(c2, up_flow * self.scaling_factor)

            # Compute the correlation
            corr = self.corr(c1, warp)
            corr = self.activation(corr)

            # Concatenate all necessary inputs
            x = torch.cat((corr, c1, up_flow, up_feat, up_classes), 1)# 81, c1.shape, 2, 2, 5

        # Apply the convolutions sequentially
        x = torch.cat((self.conv0(x), x), 1)
        x = torch.cat((self.conv1(x), x), 1)
        x = torch.cat((self.conv2(x), x), 1)

        # Predict the flow and apply deconvolution
        flow = self._predict_flow(x)
        classes = self._classify_motion(x)

        return flow, x, classes
    
if __name__ == "__main__":#TODO
    preflow = torch.randn(1, 2, 23, 30)
    preclasses = torch.randn(1, 10, 23, 30)
    
    dd = np.cumsum([64, 32, 16])
    prefeat = torch.randn(1, 16 + dd[2], 23, 30)
    
    od = 81 + 16 + 4 + 5
    block = OpticalFlowBlock(od, dd, 16 + dd[2])
    x1 = torch.randn(1, 16, 45, 60)
    x2 = torch.randn(1, 16, 90, 120)
    
    flow , feat= block(x1, x2, preflow, prefeat, preclasses)
    print(flow.shape)
    

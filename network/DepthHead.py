import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from DepthwiseConv import depthwiseConv, depthwiseDeconv
from functools import partial

class depthBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_channels=None, stride = 2):
        super(depthBlock, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        if down_channels != None:
            self.updepth = self.deconv(1, 1)
            
            self.upfeat = self.deconv(down_channels, in_channels-4, stride=stride)        
        self.conv2 = depthwiseConv(in_channels, out_channels, kernel_size = 3, norm_layer = norm_layer, activation_layer = nn.Hardswish, stride = 1)
        self.conv3 = depthwiseConv(out_channels, out_channels, kernel_size = 3, norm_layer = norm_layer, activation_layer = nn.Hardswish, stride = 1)
        self.predict_depth = depthwiseConv(out_channels, 1, kernel_size = 3, norm_layer = norm_layer, activation_layer = nn.ReLU, stride = 1)
        self.activation_layer = nn.Hardswish
        
    def deconv(self, in_planes, out_planes, kernel_size=4, stride=2, padding=1):
        return depthwiseDeconv(in_planes, out_planes, kernel_size, 1, stride, padding=padding)
    
    def forward(self, feat, flow, warped_depth, down_feat = None, down_depth = None):
        if down_feat == None:
            ones = torch.ones_like(warped_depth)
            x = torch.cat((feat, ones, flow, warped_depth), 1)
        else:
            up_feat = self.upfeat(down_feat)
            feat += up_feat
            up_depth = self.updepth(down_depth)
            x = torch.cat((feat, up_depth, flow, warped_depth), 1)
            
        x = self.conv2(x)
        x = self.conv3(x)
        
        depth = self.predict_depth(x)
        
        return x, depth
        
class depthHead(nn.Module):
    def __init__(self):
        super(depthHead, self).__init__()
        block = depthBlock
        
        fd = np.array([256, 128, 64, 32, 16])
        self.blocks = nn.ModuleList()
        self.blocks.append(block(fd[0] + 4, fd[0]))
        self.blocks.append(block(fd[1] + 4, fd[1], fd[0]))
        self.blocks.append(block(fd[2] + 4, fd[2], fd[1]))
        self.blocks.append(block(fd[3] + 4, fd[3], fd[2]))
        # self.blocks.append(block(fd[4] + 4, fd[4], fd[3]))
        
        
    def scaleMap(self, map, scale):
        # print(map.shape)
        _, _, H, W = map.shape
        rescaled_tensor = F.interpolate(map, size=(int(H * scale), int(W * scale)), mode='bilinear', align_corners=False) #TODO change the mode so that corresponding pixels won't merge
        # rescaled_tensor = rescaled_tensor.squeeze(1)
        return rescaled_tensor
    
    def forward(self, features, flow, warped_depth):
        depths = []
        with torch.no_grad():
            for i in range(4):
                scale = 1/(2**(4-i))
                scaled_flow = self.scaleMap(flow, scale)
                scaled_depth = self.scaleMap(warped_depth, scale)
                if i == 0:
                    x, depth = self.blocks[i](features[4-i], scaled_flow, scaled_depth)
                else:
                    # print(x.shape)
                    # print(depth.shape)
                    x, depth = self.blocks[i](features[4-i], scaled_flow, scaled_depth, x, depth)
                depths.append(depth)
                
        if self.training:
            return depths
        else:
            return depth

if __name__ == "__main__":
    device = 'cuda'
    
    model = depthHead()
    model.eval()
    model.to(device)
    
    warped_depth = torch.rand(1, 1, 384, 480).to(device)
    flow = torch.randn(1, 2, 384, 480).to(device)
    
    x = [
        torch.randn(1, 16, 384, 480).to(device),
        torch.randn(1, 32, 192, 240).to(device),
        torch.randn(1, 64, 96, 120).to(device),
        torch.randn(1, 128, 48, 60).to(device),
        torch.randn(1, 256, 24, 30).to(device)
    ]
    
    import time
    s = time.time()
    for i in range(100):
        depth = model(x, flow, warped_depth)
    
        print(depth.shape)
    print(100/ (time.time()-s))
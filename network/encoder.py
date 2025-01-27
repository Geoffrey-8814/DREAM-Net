import torch
import torch.nn as nn

from torchvision.models import mobilenetv3
from torchvision.ops.misc import Conv2dNormActivation
from functools import partial
from DepthwiseConv import depthwiseConv
import time

class resBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(resBlock, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        self.conv1 = depthwiseConv(in_channels, out_channels, kernel_size = 3, norm_layer = norm_layer, activation_layer = nn.Hardswish, stride = stride)
        self.conv2 = depthwiseConv(out_channels, out_channels, kernel_size = 3, norm_layer = norm_layer, activation_layer = nn.Hardswish, stride = 1)
        self.conv3 = depthwiseConv(out_channels, out_channels, kernel_size = 3, norm_layer = norm_layer, activation_layer = nn.Hardswish, stride = 1)
        self.activation_layer = nn.Hardswish
        self.identity_downsample = depthwiseConv(in_channels, out_channels, kernel_size = 1, norm_layer = norm_layer, activation_layer = nn.Hardswish, stride = stride)

    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        return x
class encoder(nn.Module):
    def __init__(self, block):
        super(encoder, self).__init__()
        self.b1 = block(3, 16, 2)
        self.b2 = block(16, 32, 2)
        self.b3 = block(32, 64, 2)
        self.b4 = block(64, 128, 2)
        self.b5 = block(128, 256, 2)
    def forward(self, x):
        x1 = self.b1(x)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        x4 = self.b4(x3)
        x5 = self.b5(x4)
        return [x1, x2, x3, x4, x5]
if __name__ == "__main__":
    _encoder = encoder(resBlock)
    _encoder.to('cuda')
    x = torch.randn(1, 3, 768, 960).to('cuda')
    s = time.time()
    for i in range(1):
        features = _encoder(x)
        for feature in features:
            print(feature.shape)
    print(1/(time.time()-s))
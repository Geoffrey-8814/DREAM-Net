import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation
from functools import partial

class depthwiseConv(nn.Module):
    def __init__(self, input_channels: int, expanded_channels: int, kernel_size: int, 
                 norm_layer, activation_layer, dilation: int = 1, stride: int = 1):
        super(depthwiseConv, self).__init__()
        
        padding = kernel_size // 2
        
        layers = []
        # expand
        if expanded_channels != input_channels:
            layers.append(
                Conv2dNormActivation(
                    input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        stride = 1 if dilation > 1 else stride
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                padding=padding
            )
        )
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)

class depthwiseDeconv(nn.Module):
    def __init__(self, input_channels: int, expanded_channels: int, kernel_size: int= 4, 
                 dilation: int = 1, stride: int = 1, padding = 1):
        super(depthwiseDeconv, self).__init__()
        
        layers = []
        # expand
        if expanded_channels != input_channels:
            layers.append(
                Conv2dNormActivation(
                    input_channels,
                    expanded_channels,
                    kernel_size=1
                )
            )

        # depthwise
        stride = 1 if dilation > 1 else stride
        layers.append(
            nn.ConvTranspose2d(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=expanded_channels,
                padding=padding
            )
        )
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)
if __name__ == "__main__":
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
    conv2d = depthwiseConv(3, 6, 3, norm_layer, nn.Hardswish, 1, 2)
    x = torch.randn(1, 3, 800, 1280)
    
    x = conv2d(x)
    
    print(x.shape)
    deconv = depthwiseDeconv(6, 6, 4, stride=2)
    
    x = deconv(x)
    
    print(x.shape)

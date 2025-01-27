# copied from https://github.com/limacv/CorrelationLayer
import torch
import torch.nn as nn

class CorrTorch(nn.Module):
    def __init__(self, pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1):
        assert kernel_size == 1, "kernel_size other than 1 is not implemented"
        assert pad_size == max_displacement
        assert stride1 == stride2 == 1
        super().__init__()
        # self.pad_size = pad_size
        # self.kernel_size = kernel_size
        # self.stride1 = stride1
        # self.stride2 = stride2
        self.max_hdisp = max_displacement
        self.padlayer = nn.ConstantPad2d(pad_size, 0)

    def forward(self, in1, in2):
        in2_pad = self.padlayer(in2)
        offsety, offsetx = torch.meshgrid([torch.arange(0, 2 * self.max_hdisp + 1),
                                           torch.arange(0, 2 * self.max_hdisp + 1)])
        hei, wid = in1.shape[2], in1.shape[3]
        output = torch.cat([
            torch.mean(in1 * in2_pad[:, :, dy:dy+hei, dx:dx+wid], 1, keepdim=True)
            for dx, dy in zip(offsetx.reshape(-1), offsety.reshape(-1))
        ], 1)
        return output
    
if __name__ == "__main__":
    corr = CorrTorch(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1)
    # Assuming you have two feature maps fmap1 and fmap2 from two images
    fmap1 = torch.randn(1, 64, 64, 256)  # Example feature map
    fmap2 = torch.randn(1, 64, 64, 256)


    result = corr(fmap1, fmap2)

    print(result.shape)
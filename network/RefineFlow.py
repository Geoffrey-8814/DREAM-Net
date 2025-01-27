import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from opticalFlowBlock import OpticalFlowBlock
import time

class refineFlow(nn.Module):
    def __init__(self):
        super(refineFlow, self).__init__()
        fd = np.array([256, 128, 64, 32, 16])
        
        self.fl = fd.size
        
        #create flow blocks
        dd = np.cumsum([16, 8, 4])
        nd = 81
        od = nd + fd[2] + 4 + 5
        up_featd = od + dd[2]
        od = 106
        self.flowBlock = OpticalFlowBlock(od, dd, up_featd, 4, 0)
            
    def scaleMap(self, map, scale):
        # print(map.shape)
        _, _, H, W = map.shape
        rescaled_tensor = F.interpolate(map, size=(int(H*scale), int(W*scale)), mode='bilinear', align_corners=False) #TODO change the mode so that corresponding pixels won't merge
        # rescaled_tensor = rescaled_tensor.squeeze(1)
        return rescaled_tensor
    def getMotion(self, K, flow, motion_classes, pre_depth, threshold, n_max):
        device = motion_classes.device
        B, D, H, W = motion_classes.shape
        masks = torch.zeros_like(motion_classes,device= device) # Initialize final depth map
        motions = torch.zeros(B, D, 4, 4,device=device)
        for b in range(B):
            motions_for_one_batch, masks_for_one_batch = self._poseEstimator(K, flow[b], motion_classes[b], pre_depth[b][0], threshold, n_max)
            if motions_for_one_batch.shape[0] != 5:#TODO
                # print(torch.zeros(5-motions_for_one_batch.shape[0], 4, 4))
                motions_for_one_batch = torch.cat((motions_for_one_batch, torch.zeros(4-motions_for_one_batch.shape[0], 4, 4)),0)
            motions[b] = motions_for_one_batch
            masks[b] = masks_for_one_batch.int()
            
        return motions, masks
    
    def forward(self, features, pre_features, flow, flow_feature, classes):
        #refine flow
        flow, _, classes = self.flowBlock(features[0], pre_features[0], flow, flow_feature, classes)

        return flow, classes
        
if __name__ == "__main__":
    model = refineFlow()
    model.eval()
    device = 'cuda'
    model.to(device)
    x = [
        torch.randn(1, 16, 384, 480).to(device),
        torch.randn(1, 32, 192, 240).to(device),
        torch.randn(1, 64, 96, 120).to(device),
        torch.randn(1, 128, 48, 60).to(device),
        torch.randn(1, 256, 24, 30).to(device)
    ]
    
    pre_features = [
        torch.randn(1, 16, 384, 480).to(device),
        torch.randn(1, 32, 192, 240).to(device),
        torch.randn(1, 64, 96, 120).to(device),
        torch.randn(1, 128, 48, 60).to(device),
        torch.randn(1, 256, 24, 30).to(device)
    ]
    flow = torch.randn(1, 2, 96, 120).to(device)
    flow_feat = torch.randn(1, 182, 96, 120).to(device)
    classes = torch.rand(1, 5, 96, 120).to(device)
    s = time.time()
    for i in range(100):
        refined_flow, refined_classes = model(x, pre_features, flow, flow_feat, classes)
        
        print(refined_flow.shape)
        print(refined_classes.shape)
    print(100/(time.time()-s))
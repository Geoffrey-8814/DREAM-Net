import torch
import torch.nn as nn
import numpy as np
from opticalFlowBlock import OpticalFlowBlock

from encoder import encoder, resBlock
import time
class backbone(nn.Module):
    def __init__(self):
        super(backbone, self).__init__()
        
        self.encoder = encoder(resBlock)
        
        dd = np.cumsum([16, 8, 4])
        fd = np.array([256, 128, 64, 32, 16])
        self.fl = fd.size
        nd = 81
        od = nd
        self.bl = 3
        self.blocks = nn.ModuleList()
        self.blocks.append(OpticalFlowBlock(od, dd))#output flow(2) feat(od + dd[2])
        for i in range(1, self.bl):
            up_featd = od + dd[2]
            od = nd + fd[i] + 4 + 5
            self.blocks.append(OpticalFlowBlock(od, dd, up_featd))
        
    def forward(self, frame, pre_features=None):
        features = self.encoder(frame)
        if pre_features == None:
            return features
        flows = []
        up_feats = []
        motionClasses = []
        for i, block in enumerate(self.blocks):
            if len(up_feats) ==0 or len(flows) == 0:
                flow, up_feat, classes = block(features[self.fl-1-i], pre_features[self.fl-1-i])
            else:
                flow, up_feat, classes = block(features[self.fl-1-i], pre_features[self.fl-1-i], flows[i - 1], up_feats[i - 1], motionClasses[i - 1])
            flows.append(flow)
            up_feats.append(up_feat)
            motionClasses.append(classes)
        if self.training:
            return flows, motionClasses
        else:
            return flows[self.bl - 1], classes, features, up_feat

if __name__ == "__main__":
    model = backbone()
    model.eval()
    model.to('cuda')
    x = torch.randn(1, 3, 480, 768).to('cuda')
    
    pre_features = model(x)
    for feature in pre_features:
        feature
    s = time.time()
    for i in range(100):
        flow, classes, _, feat = model(x, pre_features)
        print("flow", flow.shape)
        print("classes", classes.shape)
        print("feat", feat.shape)
    print(100/(time.time()-s))
    
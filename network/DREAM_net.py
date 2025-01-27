import torch
import torch.nn as nn
import numpy as np

from Backbone import backbone
from RefineFlow import refineFlow
from PoseHead import poseHead
from Projector import projector
from DepthHead import depthHead

# DREAM(Depth, flow, Reconstruction, Estimation, And Motion)
class dream_net(nn.Module):
    def __init__(self):
        super(dream_net, self).__init__()
        
        self.backBone = backbone()
        self.refineFlow = refineFlow()
        self.poseHead = poseHead()
        self.projector = projector()
        self.depthHead = depthHead()
    
    def forward(self, frame1, frame2, K, depth1):
        pre_features = self.backBone(frame1)
        flow, classes, features, flowFeat = self.backBone(frame2, pre_features)
        
        refinedFlow, refinedClasses = self.refineFlow(features, pre_features, flow, flowFeat, classes)
        
        Ts, masks = self.poseHead(K, refinedFlow, refinedClasses, depth1) 
        
        warped_depth = self.projector(depth1, K, Ts, masks)
        
        depth = self.depthHead(features, refinedFlow, warped_depth)
        
        return depth

if __name__ == "__main__":
    device = 'cuda'
    model = dream_net()
    model.eval()
    model.to(device)
    
    frame1 = torch.rand(1, 3, 480, 640).to(device)
    frame2 = torch.rand(1, 3, 480, 640).to(device)
    
    K = torch.tensor([[[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]], dtype=torch.float32).to(device)
    
    depth1 = torch.rand(1, 1, 240, 320).to(device)
    import time 
    # s = time.time()
    # for i in range(100):
    #     depth = model(frame1, frame2, K, depth1)
        
    #     print(depth.shape)
    # print(100/(time.time()-s))
    
    # Initialize the profiler
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,  # Record input shapes
        with_flops=True,  # Calculate FLOPs
        profile_memory=True  # Track memory usage
    ) as prof:
        # Run your model and record the operations
        depth = model(frame1, frame2, K, depth1)

    # Print the profiler results
    prof.export_chrome_trace("trace.json")  # You can visualize this in Chrome
    print(prof.key_averages().table(sort_by="cpu_time_total"))  # Summarize profiling by total CPU time
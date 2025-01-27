import torch
import torch.nn as nn

class projector(nn.Module):
    def __init__(self):
        super(projector, self).__init__()
        
    def depth_to_point_cloud(self, depth, K):
        """Convert depth map to 3D point cloud."""
        B, _, H, W = depth.shape
        device = depth.device

        # Create grid of pixel coordinates
        u = torch.arange(W, device=device).float()
        v = torch.arange(H, device=device).float()
        u, v = torch.meshgrid(u, v, indexing='xy')  # Shape: (H, W)
        uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1)  # Shape: (H, W, 3)
        uv1 = uv1.unsqueeze(0).expand(B, -1, -1, -1)  # Shape: (B, H, W, 3)

        # Back-project to 3D
        _depth = depth.clone()
        _depth = _depth.permute(0, 2, 3, 1)  # Shape: (B, H, W, 1)
        K_inv = torch.inverse(K)  # Shape: (B, 3, 3)
        points_3d = torch.einsum('bij,bhwj->bhwi', K_inv, uv1) * _depth  # Shape: (B, H, W, 3)    
        return points_3d

    def apply_transformations(self, points_3d, transformations, masks):
        """Apply transformations to 3D points based on masks (vectorized)."""
        B, C, H, W = masks.shape
        device = points_3d.device

        # print("p cloud",points_3d.shape)
        # Reshape points_3d to (B, H*W, 3)
        points_3d_flat = points_3d.reshape(B, H * W, 3)  # Shape: (B, H*W, 3)

        # Reshape masks to (B, C, H*W)
        masks_flat = masks.reshape(B, C, H * W)  # Shape: (B, C, H*W)

        # Initialize transformed points
        transformed_points = points_3d_flat.clone()  # Shape: (B, H*W, 3)

        # Iterate over transformations
        for c in range(C):
            # Get mask for current transformation (bool)
            mask = masks_flat[:, c]  # Shape: (B, H*W)

            # Get transformation matrix for current channel
            T = transformations[:, c]  # Shape: (B, 4, 4)

            # Apply transformation to masked points
            points_masked = points_3d_flat[mask]  # Shape: (N, 3)
            points_masked_homo = torch.cat([points_masked, torch.ones(points_masked.shape[0], 1, device=device)], dim=-1)  # Shape: (N, 4)
            points_transformed_homo = torch.einsum('bij,nj->ni', T, points_masked_homo)  # Shape: (N, 4)
            # print("points_transformed_homo", points_transformed_homo.shape)
            points_transformed = points_transformed_homo[:, :3] / (points_transformed_homo[:, 3:] + 1e-10)  # Shape: (N, 3)
            # print("points_transformed",points_transformed.shape)
            # Update transformed points
            transformed_points[mask] = points_transformed
        # print(transformed_points.shape)
        # Reshape back to (B, H, W, 3)
        transformed_points = transformed_points.reshape(B, H, W, 3)
        # print("t shape",transformed_points.shape)
        
        
        return transformed_points

    def reproject_to_depth_map(self, transformed_points, original_depth, K):
        """
        Convert transformed 3D points to a depth map, prioritizing transformed regions.
        """
        B, H, W, _ = transformed_points.shape
        device = transformed_points.device

        # Project transformed points to 2D coordinates (u, v)
        points_2d_homo = torch.einsum('bij,bhwj->bhwi', K, transformed_points)  # (B, H, W, 3)
        points_2d = points_2d_homo[..., :2] / (points_2d_homo[..., 2:] + 1e-10)  # (B, H, W, 2)
        Z_transformed = transformed_points[..., 2]  # (B, H, W)

        # Initialize depth map with original depth
        depth_map = original_depth.clone()
        # print("d",depth_map.shape)
        # Reshape indices
        B_idx = torch.arange(B, device=device)[:, None, None].expand(-1, H, W)
        u = (points_2d[..., 0].clamp(0, W-1)).long()  # (B, H, W)
        v = (points_2d[..., 1].clamp(0, H-1)).long()  # (B, H, W)
        # print(depth_map.shape)
        # print(Z_transformed.unsqueeze(1).shape)
        # Directly overwrite the depth values at (u, v) with transformed Z
        # Reshape Z_transformed to match the expected shape
        Z_transformed_reshaped = Z_transformed.unsqueeze(-1)  # Shape: (B, H, W, 1)

        # Directly overwrite the depth values at (u, v) with transformed Z
        depth_map[B_idx, :, v, u] = Z_transformed_reshaped

        return depth_map

    def project_transform_reproject(self, depth, K, transformations, masks):
        """Full pipeline: depth → 3D → transform → generate depth map."""
        # Step 1: Back-project depth to 3D points
        points_3d = self.depth_to_point_cloud(depth, K)  # (B, H, W, 3)
        # self.visualize_3d_points(points_3d, downsample_step=5)
        
        # Step 2: Apply transformations to masked regions
        transformed_points = self.apply_transformations(points_3d, transformations, masks)  # (B, H, W, 3)
        # self.visualize_3d_points(transformed_points, downsample_step=5)
        # print(tran)
        # Step 3: Reproject to depth map (overwrite original depth)
        # print(depth.shape)
        depth_transformed = self.reproject_to_depth_map(transformed_points, depth, K)

        return depth_transformed
    
    def forward(self, depth, K, T, masks):
        depth = self.project_transform_reproject(depth, K, T, masks)
        return depth

if __name__ == "__main__":
    device = 'cuda'
    model = projector()
    model.eval()
    model.to(device)
    B, C, H, W = 2, 5, 240, 320
    depth = torch.ones(B, 1, H, W)
    K = torch.tensor([[[1000, 0, 128], [0, 1000, 128], [0, 0, 1]]], dtype=torch.float32).expand(B, -1, -1)

    R = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    t =torch.tensor([0, 0, 1])
    T = torch.eye(4)
    # Set the top-left 3x3 block to the rotation matrix
    T[:3, :3] = R

    # Set the top-right 3x1 block to the translation vector
    T[:3, 3] = t
    R = T[:3, :3]  # Rotation matrix
    t = T[:3, 3:]  # Translation vector

    transformations = T.expand(B, C, -1, -1)

    # transformations = torch.rand(B, C, 4, 4)
    masks = torch.randint(1, 2, (B, C, H, W), dtype=torch.bool)

    # Run the pipeline
    # points_2d, depth_transformed = project_transform_reproject(depth, K, transformations, masks)
    depth.to(device)
    K.to(device)
    masks.to(device)
    import time
    s = time.time()
    for i in range(100):
        depth_transformed = model(depth, K, transformations, masks)
    print(100/(time.time() - s))
    # print("Reprojected 2D shape:", points_2d.shape)  # (B, H, W, 2)
    print("Transformed depth shape:", depth_transformed.shape)  # (B, 1, H, W)
    import matplotlib.pyplot as plt

    # Plot the transformed depth map for the first batch
    plt.imshow(depth_transformed[0, 0].cpu().detach().numpy(), cmap='viridis')
    plt.colorbar()
    plt.show()